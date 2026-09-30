/**
* Name: LLMAgent
* Based on the internal empty template.
* Author: dung
* Tags: LLM, intelligent agents, network communication
*
* Description: Integration module for the agents driven by LLMs (Large Language Models).
* Defines the agents that communicate with external AI systems via HTTP and WebSocket
* to take intelligent decisions in the urban transport simulation.
* Handles data synchronisation, sending observations and receiving actions.
*/


model LLMAgent

// Import of the required modules
import "Settings.gaml"
import "Inhabitant.gaml"

global {
	// HTTP connection configuration for synchronous communication
	int http_port <- 8002;
	string http_url <- "http://localhost";

	// MQTT configuration for asynchronous communication (not used currently)
	int mqtt_port <- 1883;
	string mqtt_url <- "localhost";
	string mqtt_action_topic <- "action/data";
    string mqtt_observation_topic <- "observation/data";

	init {
		// Create an HTTP synchronisation agent
		create llm_agent_sync number: 1 {
			do connect to: http_url protocol: "http" port: http_port raw: true;
		}

		// Create an asynchronous WebSocket communication agent
		create llm_agent_async number: 1 {
			do connect protocol: "websocket_server" port: 3001 with_name: name raw: true;
		}
	}
}

/**
 * LLM synchronisation agent - handles the periodic communication with the LLM system via HTTP.
 * Sends synchronisation data every 15 minutes and population data every hour.
 * Responsible for the initialisation of the population and for continuous synchronisation.
 */
species llm_agent_sync skills:[network] {
	/**
	 * Initialisation - sends the initialisation data to the LLM system at the first cycle
	 */
	reflex init when: cycle = 1 {
		write "Init population -> LLM, timestamp: " + CURRENT_TIMESTAMP;
		write "Parameters sent: population=" + population_size
			+ " part_of_llm=" + part_of_llm_based_agents
			+ " ltm=" + long_term_memory_enabled
			+ " self_reflect=" + long_term_self_reflect_enabled
			+ " max_days=" + simulation_max_days
			+ " accidents=" + accidents_enabled;

		do send to: "/init" contents: [
			"POST",
			to_json([
				"timestamp"::CURRENT_TIMESTAMP,
				"population_size"::population_size,
				"part_of_llm_based_agents"::part_of_llm_based_agents,
				"long_term_memory_enabled"::long_term_memory_enabled,
				"long_term_self_reflect_enabled"::long_term_self_reflect_enabled,
				// Stop horizon (ticket 008, A5): sent to be recorded in
				// the run's scenario_params.yaml. Without it, nothing in the experiment
				// directory says over how many days the run was supposed to span.
				"simulation_max_days"::simulation_max_days,
				"prefixe_commun"::prefixe_commun,
				// Accidents on the roads (ticket 070): the experimenter alone decides,
				// and the controller records it in the run's scenario_params.yaml.
				"accidents_enabled"::accidents_enabled
			]),
			["Content-Type"::"application/json"]
		];
	}

	/**
	 * Periodic synchronisation - sends the population counters every 15 minutes
	 * Activity ends are handled via the WebSocket observations (submit_obseration)
	 */
	reflex sync when: cycle > 1 and every(15#mn) {
		int nb_ready    <- length(inhabitant where (each.is_ready));
		int nb_active   <- length(inhabitant where (each.is_active));
		int nb_inactive <- length(inhabitant) - nb_ready - nb_active;

		string json_body <- to_json(["timestamp"::CURRENT_TIMESTAMP,
			"ready_count"::nb_ready, "active_count"::nb_active, "inactive_count"::nb_inactive]);
		if prefixe_commun {
			string attendu <- "synchronized:" + string(CURRENT_TIMESTAMP);
			string en_cours <- "pending:" + string(CURRENT_TIMESTAMP);
			PREFIXE_SYNC_REPONSE <- "";
			PREFIXE_SYNC_ERREUR <- false;
			int max_retries <- 5;
			int current_retry <- 0;
			loop while: PREFIXE_SYNC_REPONSE != attendu and !PREFIXE_SYNC_ERREUR {
				do send to: "/sync" contents: [
					"POST", json_body, ["Content-Type"::"application/json"]
				];
				// The controller may answer « pending » after 24 s. The SAME
				// timestamp is resent, without advancing the simulation clock.
				int wait_attempts <- 0;
				loop while: (PREFIXE_SYNC_REPONSE != attendu) and
					(PREFIXE_SYNC_REPONSE != en_cours) and (!PREFIXE_SYNC_ERREUR) and (wait_attempts < 50) {
					do traiter_messages_http;
					if (PREFIXE_SYNC_REPONSE != attendu) and
						(PREFIXE_SYNC_REPONSE != en_cours) and (!PREFIXE_SYNC_ERREUR) {
						do fetch_message_from_network;
					}
					wait_attempts <- wait_attempts + 1;
				}
				if PREFIXE_SYNC_REPONSE = en_cours {
					PREFIXE_SYNC_REPONSE <- "";
					current_retry <- 0;
				} else if (PREFIXE_SYNC_REPONSE != attendu) and (!PREFIXE_SYNC_ERREUR) {
					current_retry <- current_retry + 1;
					write "[WARN] /sync acknowledgement not received for timestamp=" + string(CURRENT_TIMESTAMP) + " (attempt " + current_retry + "/" + max_retries + ")";
					if current_retry >= max_retries {
						PREFIXE_SYNC_ERREUR <- true;
					}
				}
			}
			if PREFIXE_SYNC_ERREUR {
				write "[ALARME] Préfixe commun : /sync failed after retries, simulation à invalider.";
			}
		} else {
			do send to: "/sync" contents: [
				"POST", json_body, ["Content-Type"::"application/json"]
			];
		}
	}
	
	
	/**
	 * Reception and processing of the messages from the LLM system
	 * Processes the initialisation responses and creates the agent population
	 */
	action traiter_messages_http {
		loop while:has_more_message()
		{
			message mess <- fetch_message();
			string jsonBody <- map(mess.contents)["BODY"];
			// Guard against non-JSON responses (e.g. HTTP 500 "Internal Server Error")
			if jsonBody = nil or not (jsonBody contains "{") {
				write "[ERROR] Received non-JSON HTTP response from controller: " + jsonBody;
				if prefixe_commun { PREFIXE_SYNC_ERREUR <- true; }
				continue;
			}
			map<string, unknown> json <- from_json(jsonBody);
			if bool(json["success"]) != true {
				write "[ERROR] Got error message: " + string(json);
				if prefixe_commun { PREFIXE_SYNC_ERREUR <- true; }
				continue;
			}
			string messageType <- json["message_type"];
			if messageType = "ag_sync" {
				PREFIXE_SYNC_REPONSE <- string(json["data"]);
				continue;
			}
			
			/** 
			 *   --------   WORLD INIT   --------------
			 */
			if messageType = "ag_world_init" {
				// Process the initialisation of the agents' world
				map<string, unknown> data <- json["data"];
				list<map<string, unknown>> people <- data["people"];
				loop p over: people {
					map<string, unknown> p_loc <- map<string, unknown>(p["location"]);
					float lon <- float(p_loc["lon"]);
					float lat <- float(p_loc["lat"]);
					point plocation <- point(to_GAMA_CRS({lon, lat}, POPULATION_CRS));
					create inhabitant with: [
						route_vehicle_map::ROUTE_VEHICLE_MAP,
						person_name::string(p["name"]),
						person_id::string(p["person_id"]),
//						age::int(p["age"]),
						location::plocation,
						is_llm_based::bool(p["is_llm_based"])
					] {
						INHABITANT_MAP[self.person_id] <- self;
					}
				}
				// Ticket 031 (G1): no agent may be outside the GAMA world. Counted and logged
				// once, at creation — it is the run's « zero agent outside the GAMA world » measurement.
				int hors_monde <- length(inhabitant where (!(world.shape covers each.location)));
				write "[PERIMETRE] " + length(people) + " inhabitants created, " + hors_monde + " outside the GAMA world.";
				if (hors_monde > 0) {
					write "[ALARME] " + hors_monde + " inhabitant(s) outside the GAMA world: the perimeter of the population and the footprint of the world (perimetre_453.shp) do not coincide.";
				}
			} else if messageType = "calibration_started" {
				// Start acknowledgement of the prompt calibration
				map<string, unknown> data <- json["data"];
				write "[CALIBRATION] Démarrée (pid=" + string(data["pid"])
					+ ", cycles=" + string(data["iterations"])
					+ ") — journal : " + string(data["log"]);
			}
		}

	}

	reflex get_message when: has_more_message() {
		do traiter_messages_http;
	}

	/**
	 * Launches the prompt calibration on the controller side (POST /calibrate).
	 * Non-blocking: the controller runs the campaign as a background task.
	 * The number of cycles (loop iterations) comes from the global
	 * parameter `calibration_cycles`, adjustable from the GUI.
	 */
	/**
	 * Places a CHOSEN accident on the controller side (POST /accidents), on the graph edge
	 * closest to the point set in the GUI, at the given hour of the current simulated day.
	 * Refused — and logged as such — if the accident regime is unticked.
	 */
	action poser_accident {
		// Midnight of the current simulated day, plus the requested hour.
		int _minuit <- CURRENT_TIMESTAMP - (CURRENT_TIMESTAMP mod 86400);
		int _debut <- _minuit + accident_pose_heure * 3600;
		write "[ACCIDENT] Pose demandée en (" + accident_pose_lat + ", " + accident_pose_lon
			+ ") à " + accident_pose_heure + " h pour " + accident_pose_duree + " min...";
		do send to: "/accidents" contents: [
			"POST",
			to_json([
				"lat"::accident_pose_lat,
				"lon"::accident_pose_lon,
				"debut_ts"::_debut,
				"duree_minutes"::accident_pose_duree
			]),
			["Content-Type"::"application/json"]
		];
	}

	action launch_calibration {
		write "[CALIBRATION] Launch request — " + calibration_cycles + " cycle(s)...";
		do send to: "/calibrate" contents: [
			"POST",
			to_json(["iterations"::calibration_cycles]),
			["Content-Type"::"application/json"]
		];
	}
}


species llm_agent_async skills:[network] {
	string send_to;  // Identifier of the WebSocket recipient

//	reflex send when: send_to != nil and every(2#mn) {
//		write "Sending...";
//		do send to: send_to contents: name + " at " + cycle + " sent to server_group a message";
//	}

	/**
	 * Submission of observations - sends the observations collected by the inhabitant agents
	 * Every 5 minutes, transmits the observation data for the LLM's learning
	 */
	reflex submit_obseration when: send_to !=nil and every(1#cycle) {
		loop p over: (inhabitant where (length(each.OB_LIST) > 0)) {
			list<map<string, unknown>> ob_list <- p.OB_LIST;
			p.OB_LIST <- [];
			loop ob over: ob_list {
				point ploc <- point(p.location CRS_transform(POPULATION_CRS));
				map<string, unknown> ob_payload <- [
					"person_id"::p.person_id,
					"activity_id"::ob["activity_id"],
					"timestamp"::CURRENT_TIMESTAMP,
					"location"::[
						"lon"::ploc.x,
			    		"lat"::ploc.y
					],
				    "env_ob_code"::string(ob["type"]),
				    "data"::ob
				];
				string payload <- to_json([
					"topic"::"observation/data",
					"payload"::ob_payload
				]);
				do send to: send_to contents: payload;
				//write "Send observation of " + p.person_id + ": " + ob;
			}
		}
	}
	   	
	/**
	 * Reception of the actions from the LLM system - processes the incoming WebSocket messages
	 * Receives the LLM's action decisions and applies them to the appropriate inhabitant agents
	 */
	reflex get_message when: has_more_message() {
		loop while:has_more_message()
		{
			message mess <- fetch_message();
			send_to <- mess.sender;  // Remember the sender for the replies
			//write "mess.contents " + map(mess.contents);
			string action_data_json <- map(mess.contents)["contents"];
			map<string, unknown> payload_data <- from_json(action_data_json);
			string topic <- payload_data["topic"];
			
			/** 
			 *   --------   LOG   --------------
			 */
			if topic = "system/log" {
				map<string, unknown> log_payload <- map<string, unknown>(payload_data["payload"]);
				write "[Python] " + string(log_payload["message"]);
				continue;
			}

			/**
			 *   --------   THROTTLE (predictive backpressure, ticket 003)   --------------
			 * Degraded regime reported by Python: actual LLM rate and simulation speed.
			 * The `message` field stays self-contained (treated as a log); the globals
			 * feed the experiment UI (monitor/overlay).
			 */
			if topic = "system/throttle" {
				map<string, unknown> t <- map<string, unknown>(payload_data["payload"]);
				THROTTLE_ACTIVE  <- bool(t["active"]);
				LLM_RATE_PER_MIN <- float(t["llm_rate_per_min"]);
				SIM_RATIO_PYTHON <- float(t["sim_ratio"]);
				write "[Python][throttle] " + string(t["message"]);
				continue;
			}

			/** 
			 *   --------   ACTION/DATA   --------------
			 */
			if topic != "action/data" {
				continue;
			}
			map<string, unknown> action_data <- payload_data["payload"];
			

			string person_id <- action_data["person_id"];
			map<string, unknown> data <- action_data["action"];
			inhabitant person <- INHABITANT_MAP[person_id];
			if person != nil {
				// Apply the action to the agent found
				ask person {
					
					// Moving ID (serialized as "id" by PersonMove.model_dump())
					self.moving_id <- string(data["id"]);
					
					// Purpose of the trip (e.g. going to work from 9 a.m. to 6 p.m.)
					map<string, unknown> for_activity <- map<string, unknown>(data["for_activity"]);
					self.activity_id <- string(for_activity["id"]);
					self.purpose <- string(data["purpose"]);
					self.expected_arrive_at <- int(data["expected_arrive_at"]);
					int prepare_before_seconds <- int(data["prepare_before_seconds"]);
					self.schedule_at <- self.expected_arrive_at - prepare_before_seconds;
					//  write "[Plan] Person " + person_id + " purpose=" + string(data["purpose"]) + " expected_arrive_at=" + self.expected_arrive_at + " prepare_before_seconds=" + prepare_before_seconds + " schedule_at=" + self.schedule_at + " (now=" + CURRENT_TIMESTAMP + ")";
					if CURRENT_TIMESTAMP > self.schedule_at {
						int sched_h <- int((self.schedule_at mod SECONDS_IN_24H) / 3600);
						int sched_m <- int(((self.schedule_at mod SECONDS_IN_24H) mod 3600) / 60);
						string formatted_sched <- "" + sched_h + "h" + (sched_m < 10 ? "0" : "") + sched_m;
						int now_h <- int(CURRENT_TIMESTAMP_24H / 3600);
						int now_m <- int((CURRENT_TIMESTAMP_24H mod 3600) / 60);
						string formatted_now <- "" + now_h + "h" + (now_m < 10 ? "0" : "") + now_m;
						write "⚠️ Late order: Person " + person_id + " received order to start trip at " + formatted_sched + " when current time is " + formatted_now;
					}
					//	self.moving_description <- string(data["description"]);
						
					// Definition of the travel plan
					//map<string, unknown> plan <- map<string, unknown>(data["plan"]);
					map plan_map <- map(data["plan"]);
					list legs_to_send <- list(plan_map["legs"]);
					
					do passenger_set_plan(
						data["target_location"],
						legs_to_send,
						data
					);
				}	
			} else {
				 write "[LLM Message: action/data] Not found the person: " + person_id;
			}
			
		}
		
	}
}

/**
 * Test LLM agent - used to debug and test network communication.
 * Simply displays all the messages received, to check that it works.
 */
species llm_agent_test skills:[network] {
	/**
	 * Test reception - displays all the messages received for debugging
	 */
	reflex get_message {
		loop while:has_more_message()
		{
			message mess <- fetch_message();
			write "mess " + mess;
		}
		
	}
}
