/**
* Name: People (Inhabitants)
* Based on the internal empty template.
* Author: dung
* Tags: agents, mobility, transport, passengers
*
* Description: Human agents (inhabitants) who move around the city using public transport.
* This model defines the behaviour of people navigating the urban transport system,
* including walking, waiting for vehicles, transit rides, and activity planning.
* Supports both regular agents and intelligent agents driven by LLMs.
*/


model People

// Import of the public transport system for interactions with vehicles
import "PublicTransport.gaml"


global {
    // Display parameters for the inhabitants
    float inhabitant_display_size <- 20.0;           // Display size of the inhabitants
    bool show_inhabitants <- true;                   // Show the inhabitants
    int show_inhabitants_label_density <- 1;       // Percentage of agents showing labels

    // Global registry of all the inhabitants for fast lookup
    map<string, inhabitant> INHABITANT_MAP <- [];
}


/* Insert your model definition here */

/**
 * Base species for the agents that can move around the city.
 * Provides the fundamental movement capabilities and distance tracking.
 * Virtual species - serves as parent for the concrete mobile agents.
 */
species in_transfer skills: [moving] virtual: true {
    // Movement state
    point moving_target;                    // Current movement destination
    bool is_stop_moving -> moving_target = nil;  // True when not moving

    // Distance tracking for the metrics
    float last_dist_traveled <- 0.0;       // Total distance travelled in the current segment

    // Movement parameters
    float speed <- 2#m/#s;                 // Walking speed (2 m/s)
    float moving_close_dist <- 15#m;       // Distance threshold to consider the destination reached

    /**
     * Continuous movement reflex - moves the agent towards the target
     */
    reflex moving_update when: !is_stop_moving {
        // TODO: move along the roads extracted from the OSM data
        do goto target: moving_target speed: speed;
        last_dist_traveled <- last_dist_traveled + real_speed * step;
        
        // Check whether the destination is reached
        if (location distance_to moving_target < moving_close_dist) {
            location <- moving_target;
            moving_target <- nil;
        }
    }

    /**
     * Reset the travelled distance metrics
     */
    action metrics_reset_dist_traveled {
        last_dist_traveled <- 0.0;
    }
}

/**
 * Passenger species - agents that use public transport.
 * Extends in_transfer with trip planning, boarding/alighting vehicles,
 * and the handling of multimodal journeys.
 * Virtual species - serves as parent for the concrete passenger implementations.
 */
species passenger parent: in_transfer virtual: true {
    // Activity state
    bool is_ready <- false;               // True when the agent has an active trip plan
    bool is_active <- false;               // True when the agent is moving

    // Fast lookup cache for the route vehicles
    map<string, list<public_vehicle>> route_vehicle_map;

    // Trip planning parameters
    string moving_id;                      // Unique identifier of the trip
    string activity_id;                    // Associated activity identifier
    string purpose;                        // Purpose of the trip (work, home, leisure, etc.)
    int expected_arrive_at;               // Expected arrival timestamp
    int schedule_at;                       // Scheduled departure timestamp
    map<string, unknown> raw_trip;         // Raw trip data from the planning system
    string moving_description;             // Readable description of the trip
    point target_location -> length(list_destination) > 0 ? list_destination[length(list_destination)-1]: nil;

    // Route constants
    string _WALK_ <- "__DIRECT_FOOT__";           // Marker for the walking segments
    string _CAR_  <- "__DIRECT_CAR__";      // Marker for the car segments
    string _BIKE_  <- "__DIRECT_BICYCLE__";          // Marker for the bike segments
    string _PUBLIC_  <- "__PUBLIC__";      // Marker for public transport

    // Interaction state with the vehicles
    public_vehicle on_vehicle;             // Vehicle currently boarded (nil if walking)
    float get_in_vehicle_dist <- 25#m;     // Distance threshold for boarding vehicles
    int step_idx <- 0;                     // Current step in the travel plan

    // Data structures of the travel plan
    list<point> list_destination <- [];                    // Geographic destinations
    list<string> list_destination_stop_name <- [];         // Stop names for each destination
    list<string> list_route_id <- [];                      // Route IDs (or _ROUTE_NONE_ for walking)
    list<list<string>> list_shape_id <- [];                // Shape IDs for the transit segments
    list<float> list_planned_step_duration <- [];          // Planned duration (seconds) per step
    float default_car_speed <- 50#km/#h;                   // Default car speed ~50 km/h
    string main_mode <- "";

    // Metrics tracking
    int step_started_at <- 0;                              // Timestamp of the start of the current step
    // Ticket 077, lot B1 — the plan is RECEIVED well before the agent leaves: it waits for
    // `schedule_at`. As long as this flag is false, the step stopwatch is meaningless, and
    // setting it on reception made the first segment absorb the whole wait. Measured
    // on the run of ticket 075: 398 walking observations out of 412 had a ZERO distance
    // and a duration exactly equal to the departure delay. Motorised agents then read themselves
    // back as walkers of five to twelve hours.
    bool plan_started <- false;                            // Has the trip ACTUALLY started
    float on_vehicle_capacity_utilization <- 0.0;          // Vehicle capacity at boarding
    float trip_traveled_duration <- 0.0;                   // Total trip duration so far

    // Virtual actions for metrics collection (to be implemented by the subclasses)
    action submit_ob_transfer(float segment_duration, float dist, int ob_step_idx) virtual: true;
    action submit_ob_transit(float segment_duration, float dist, int ob_step_idx, float capacity) virtual: true;
    action submit_ob_vehicle(float segment_duration, float dist, string vehicle_mode) virtual: true;
    action submit_ob_tripfeedback(float trip_duration) virtual: true;
    action submit_vehicle_wait_time(float wait_duration, int ob_step_idx) virtual: true;
    action submit_ob_tc_timeout(float wait_duration, int ob_step_idx) virtual: true;

    // Readable identifier of the person (e.g. "519453")
    string person_id <- "";

    // Activity counter
    int total_activities <- 0;

    /**
     * Reset the current trip plan - clear all the destinations and routes
     */
    action passenger_reset_plan {
        list_destination <- [];
        list_destination_stop_name <- [];
        list_route_id <- [];
        list_shape_id <- [];
        list_planned_step_duration <- [];
        main_mode <- "";
    }

    /**
     * Define a new trip plan for the agent based on the travel planning data.
     * Converts the raw trip data into executable travel segments.
     *
     * @param plan_target Coordinates of the final destination {lon, lat}
     * @param legs_raw List of the trip legs from the routing engine
     * @param raw Raw trip data structure
     */
    action passenger_set_plan(map plan_target, list legs, map raw) {
        
        total_activities <- total_activities + 1;
				
        // Handle direct teleportation for empty legs (no public transport needed)
        if length(legs) = 0 {
            map raw_loc <- (raw != nil and raw.keys contains "plan" and raw["plan"] is map) ? map(raw["plan"]) : nil;
            map start_loc <- (raw_loc != nil and raw_loc.keys contains "start_location" and raw_loc["start_location"] is map) ? map(raw_loc["start_location"]) : nil;
            
            if (start_loc != nil) {
                if (!(start_loc.keys contains "lon" or start_loc.keys contains "lng") or !(start_loc.keys contains "lat")) {
                    write "❌ ERROR: Missing coordinates in start_location for agent " + name + "! Data: " + start_loc;
                } else {
                    float start_lon <- start_loc.keys contains "lon" ? float(start_loc["lon"]) : float(start_loc["lng"]);
                    float start_lat <- float(start_loc["lat"]);
                    point start_point <- point(to_GAMA_CRS({start_lon, start_lat}, POPULATION_CRS));
                    location <- start_point;
                }
            } else {
                write "❌ ERROR: start_location is nil for agent " + name + " !";
            }
        }

        // Activate the agent only if there is a plan to follow.
        // legs=[] = same location: no movement possible, is_ready stays false
        // to avoid the deadlock (agent stuck "ready" without ever generating an "arrival").
        is_ready <- length(legs) > 0;
        raw_trip <- raw;

        // Reset the trip state and the metrics
        step_idx <- 0;
        trip_traveled_duration <- 0.0;
        step_started_at <- CURRENT_TIMESTAMP;
        // The stopwatch will be RESET at the first real movement (ticket 077, lot B1). The value
        // above stays a fallback: if the agent leaves without waiting, the two coincide.
        plan_started <- false;

        do passenger_reset_plan();

        // Build the travel plan from the routing legs
        if length(legs) > 0 {
            // Add the initial walking segment towards the first transit stop
            map leg0 <- map(legs[0]);
            map start_loc_0 <- map(leg0["start_location"]);

            if (!(start_loc_0.keys contains "lon" or start_loc_0.keys contains "lng") or !(start_loc_0.keys contains "lat")) {
                write "❌ ERROR: Missing coordinates in legs[0]['start_location'] for agent " + name + "! Data: " + start_loc_0;
            } else {
                float start_lon <- start_loc_0.keys contains "lon" ? float(start_loc_0["lon"]) : float(start_loc_0["lng"]);
                float start_lat <- float(start_loc_0["lat"]);

                point start_point <- point(to_GAMA_CRS({start_lon,start_lat}, POPULATION_CRS));
                list_destination << start_point;
                list_destination_stop_name << string(start_loc_0["stop"]);
                list_route_id << _WALK_;
                list_shape_id << nil;
                list_planned_step_duration << 0.0;
            }

            // Process each transit leg
            loop leg over: legs {
                map leg_map <- map(leg);
                map leg_end_location <- map(leg_map["end_location"]);
                
                if (!(leg_end_location.keys contains "lon" or leg_end_location.keys contains "lng") or !(leg_end_location.keys contains "lat")) {
                    write "❌ ERROR: Missing coordinates in leg['end_location'] for agent " + name + "! Data: " + leg_end_location;
                } else {
                    float leg_end_lon <- leg_end_location.keys contains "lon" ? float(leg_end_location["lon"]) : float(leg_end_location["lng"]);
                    float leg_end_lat <- float(leg_end_location["lat"]);

                    point end_point <- point(to_GAMA_CRS({leg_end_lon, leg_end_lat}, POPULATION_CRS));
                    list_destination << end_point;
                    list_destination_stop_name << string(leg_end_location["stop"]);

                    // Determine whether it is a transfer (walking) or a transit segment
                    string transit_route <- string(leg_map["transit_route"]);
                    string mode <- (bool(leg_map["is_transfer"]) ? _WALK_: string(leg_map["transit_route"]));
                    list_route_id << mode;
                    list_shape_id << (bool(leg_map["is_transfer"]) ? nil : (list(leg_map["shape_id"]) collect string(each)));
                    
                    // Segment duration: "duration" field (seconds) or computed from end_time-start_time (ms)
                    float _leg_dur <- (leg_map.keys contains "duration" and leg_map["duration"] != nil) ?
                        float(int(leg_map["duration"])) :
                        (float(int(leg_map["end_time"])) - float(int(leg_map["start_time"]))) / 1000.0;
                    list_planned_step_duration << _leg_dur;
                    
                    // Update the main mode if undefined or walking.
                    if main_mode = "" {
                    	main_mode <- mode;
                    }
                    else if main_mode = _WALK_ {
                    	main_mode <- mode;
                    }
                }
            }
        }

        // Add the final walking segment towards the destination
        if (!(plan_target.keys contains "lon" or plan_target.keys contains "lng") or !(plan_target.keys contains "lat")) {
            write "❌ ERROR: Missing coordinates in the final destination for agent " + name + "! Data: " + plan_target;
        } else {
            float target_lon <- plan_target.keys contains "lon" ? float(plan_target["lon"]) : float(plan_target["lng"]);
            point end_point <- point(to_GAMA_CRS({target_lon, float(plan_target["lat"])}, POPULATION_CRS));
            
            list_destination << end_point;
            list_destination_stop_name << purpose;
            list_route_id << _WALK_;
            list_shape_id << nil;
            list_planned_step_duration << 0.0;
        }
    }

    /**
     * Virtual action called when the trip plan is finished
     * To be implemented by the subclasses for a specific end behaviour
     */
    action on_finish_plan virtual: true {

    }
	
//	reflex follow_the_vehicle when: on_vehicle != nil {
//		if !dead(on_vehicle) {
//			// follow the vehicle if we're sitting on it
//			location <- on_vehicle.location;
//		}
//		else {
//			point dest <- list_destination[step_idx];
//			location <- dest;
//			on_vehicle <- nil;
//		}
//		
////		// get off if we reach to the last stop, or close to the destination
////		point dest <- list_destination[step_idx];
////		if location distance_to dest <= get_in_vehicle_dist or dead(on_vehicle){
////			if !dead(on_vehicle) {
////				ask on_vehicle {
////					do get_off(name);
////				}
////			}
////			on_vehicle <- nil;
////			location <- dest;
////		}
//	}
	
		
	reflex follow_the_vehicle when: on_vehicle != nil {
		if CURRENT_TIMESTAMP < schedule_at {
			return;
		}
		
		if !dead(on_vehicle) {
			// follow the vehicle if we are sitting on it
			location <- on_vehicle.location;
		}
		
		// get off if we reach the last stop, or close to the destination
		point dest <- list_destination[step_idx];
		if location distance_to dest <= get_in_vehicle_dist or dead(on_vehicle){
			if !dead(on_vehicle) {
				ask on_vehicle {
					do get_off(name);
				}
				// metrics
				on_vehicle_capacity_utilization <- on_vehicle.capacity_utilization;
			}		
			on_vehicle <- nil;
			location <- dest;
		}
		
	}
	
	reflex follow_the_plan_when_stop when: target_location != nil and is_stop_moving and on_vehicle = nil {
		
		/* If the current time is earlier than the scheduled time (schedule_at), the agent does nothing and waits.  */
		if CURRENT_TIMESTAMP < schedule_at {
			return;
		}
		
		 is_ready <- false;
		 is_active <- true;

		// Ticket 077, lot B1 — the agent has just passed `schedule_at`: it is NOW that
		// the trip starts. Without this reset, the first segment carries the wait that
		// preceded it, and the agent's memory reads it as walking time.
		if !plan_started {
			plan_started <- true;
			step_started_at <- CURRENT_TIMESTAMP;
		}

		point dest <- list_destination[step_idx];
		
		// move to the next step if the destination of the previous step is reached
		if location distance_to dest < moving_close_dist {
			// try to submit the observation
			string _route <- list_route_id[step_idx];
			bool is_transfer <- _route = _WALK_;
			// Ticket 077, lot B2 — a PERSONAL vehicle is neither walking nor public
			// transport. It used to go down the transit branch, whose template
			// queries the GTFS: 253 car trips of the 075 run were rendered to the
			// model as « Trip by Unknown Unknown » between two unnamed stops.
			bool is_own_vehicle <- _route = _CAR_ or _route = _BIKE_;
			float _duration <- float(CURRENT_TIMESTAMP-step_started_at);
			// metrics
			trip_traveled_duration <- trip_traveled_duration + _duration;

			if is_transfer {
				do submit_ob_transfer(
					_duration,
					last_dist_traveled,
					step_idx
				);
			} else if is_own_vehicle {
				do submit_ob_vehicle(
					_duration,
					last_dist_traveled,
					_route = _CAR_ ? "car" : "bike"
				);
			} else {
				do submit_ob_transit(
					_duration,
					last_dist_traveled,
					step_idx,
					on_vehicle_capacity_utilization
				);
			}
			step_idx <- step_idx + 1;
			location <- dest;
			
			// reset the metrics
			step_started_at <- CURRENT_TIMESTAMP;
			last_dist_traveled <- 0.0;
		}
		
//		write "Stop: " + step_idx;
		
		/* If the step index goes beyond the list of destinations, the journey is finished */
		if step_idx >= length(list_destination) {
			location <- target_location;
			
			do submit_ob_tripfeedback(trip_traveled_duration);
			
			do passenger_reset_plan();
			do on_finish_plan();
			
			activity_id <- nil;
			is_active <- false;
			return;
		}
		
		// plan the next movement, own movement or waiting for a vehicle
		string route_id <- list_route_id[step_idx];
		list<string> shape_id_list <- list_shape_id[step_idx];
		
		/* CAR movement case — speed = distance / planned duration of the segment */
		if route_id = _CAR_ or route_id = _BIKE_ {
			point dest2 <- list_destination[step_idx];
			float planned_dur <- (step_idx < length(list_planned_step_duration)) ? list_planned_step_duration[step_idx] : 0.0;
			float car_dist <- location distance_to dest2;
			speed <- (planned_dur > 0 and car_dist > 0) ? (car_dist / planned_dur) : default_car_speed;
			moving_target <- dest2;
		}
		/* Walking movement case */
		else if route_id = _WALK_{
			speed <- 2#m/#s;
			point dest2 <- list_destination[step_idx];
			moving_target <- dest2;
		}
		/* PUBLIC transport case */
		else {
			int MAX_VEHICLE_WAIT <- 30 * 60;
			int wait_so_far <- CURRENT_TIMESTAMP - step_started_at;

			if route_id in route_vehicle_map.keys {
				// TODO: take the vehicle capacity into account
				public_vehicle closest_vehicle <- (route_vehicle_map[route_id]
						first_with (shape_id_list contains each.shape_id and !each.is_full and distance_to(each, self) < get_in_vehicle_dist)
				);
				if closest_vehicle != nil {
					on_vehicle <- closest_vehicle;
					ask closest_vehicle {
						do get_in(name);
					}

					float waiting_duration <- float(CURRENT_TIMESTAMP-step_started_at);
					do submit_vehicle_wait_time(waiting_duration, step_idx);

					// metrics
					on_vehicle_capacity_utilization <- on_vehicle.capacity_utilization;
				} else {
					if wait_so_far > MAX_VEHICLE_WAIT {
						do submit_ob_tc_timeout(float(wait_so_far), step_idx);
						location <- target_location;
						do passenger_reset_plan();
						do on_finish_plan();
						activity_id <- nil;
						is_active <- false;
					}
				}
			}
			else {
				if wait_so_far > MAX_VEHICLE_WAIT {
					do submit_ob_tc_timeout(float(wait_so_far), step_idx);
					location <- target_location;
					do passenger_reset_plan();
					do on_finish_plan();
					activity_id <- nil;
					is_active <- false;
				}
			}
		}
	}
}

/**
 * Concrete inhabitant species - represents the individual people in the simulation.
 * Extends passenger with identity, LLM integration, and observation collection.
 * It is the main agent type users interact with in the simulation.
 */
species inhabitant parent: passenger {
    // Identity and personal attributes
    string person_name;                    // Full name
    bool is_llm_based <- false;            // Whether this agent uses an LLM for decisions

    // Activity state
    int time_24h -> CURRENT_TIMESTAMP_24H; // Current hour in 24h format
    bool is_idle -> target_location = nil; // True when the agent has no active trip

    // Observation collection for the LLM agents
    list<map<string,unknown>> OB_LIST <- [];  // List of observations for learning

    // Display parameters
    bool show_name <- flip(show_inhabitants_label_density/100.0);  // Whether to show the name label

    /**
     * Initialise the inhabitant with a default purpose
     */
    init {
        purpose <- "home";
    }

    /**
     * Called when the trip plan is finished
     * Logs the end and could notify the LLM agent
     */
    action on_finish_plan {

    	// Computation of the absolute duration in seconds
	    float diff_seconds <- float(abs(CURRENT_TIMESTAMP - expected_arrive_at));
		// Bring the Unix timestamps back to the time of day (mod 24h) before formatting
		float expected_arrive_at_seconds <- float(expected_arrive_at mod SECONDS_IN_24H);
		float schedule_at_seconds <- float(schedule_at mod SECONDS_IN_24H);

	    // Extraction of the units
	    int diff_h <- int(diff_seconds / 3600);
	    int diff_m <- int((diff_seconds mod 3600) / 60);
	    string formatted_diff_time <- "" + diff_h + "h" + (diff_m < 10 ? "0" : "") + diff_m;

		int expected_arrive_h <- int(expected_arrive_at_seconds / 3600);
	    int expected_arrive_m <- int((expected_arrive_at_seconds mod 3600) / 60);
	    string formatted_expected_arrive_time <- "" + expected_arrive_h + "h" + (expected_arrive_m < 10 ? "0" : "") + expected_arrive_m;

		int schedule_h <- int(schedule_at_seconds / 3600);
	    int schedule_m <- int((schedule_at_seconds mod 3600) / 60);
	    string formatted_schedule_time <- "" + schedule_h + "h" + (schedule_m < 10 ? "0" : "") + schedule_m;

	    int current_h <- int(CURRENT_TIMESTAMP_24H / 3600);
	    int current_m <- int((CURRENT_TIMESTAMP_24H mod 3600) / 60);
	    string formatted_current_time <- "" + current_h + "h" + (current_m < 10 ? "0" : "") + current_m;

		string delay_msg <- ": traject schedule at="+formatted_schedule_time+ " expected arrive at=" + formatted_expected_arrive_time + " current time is " + formatted_current_time;
	    if (CURRENT_TIMESTAMP <= expected_arrive_at) {
	        write "Hura 😊, Person " + person_id + " finished the plan with " + formatted_diff_time + " in advance"+delay_msg; 
	    } 
	    else if (diff_seconds <= 60*15) {
	        write "Too late 😡, Person " + person_id + " finished the plan " + formatted_diff_time + " late"+delay_msg; 
	    }
	    else {
	        write "Too late 🤬, Person " + person_id + " finished the plan " + formatted_diff_time + " very late"+delay_msg; 
	    }

    }
	
	/**
	 * Submit an observation for a walking/transfer segment
	 * Records the duration, the distance and the stop information for learning
	 */
	action submit_ob_transfer(float segment_duration, float dist, int ob_step_idx) {
		map<string,unknown> ob <- [
			"type"::"transfer",
			"timestamp"::CURRENT_TIMESTAMP,
			"moving_id"::moving_id,
			"activity_id"::activity_id,
			"distance"::dist,
			"duration"::segment_duration,
			"from_name"::(ob_step_idx = 0? nil: list_destination_stop_name[ob_step_idx-1]),
			"destination_name"::list_destination_stop_name[ob_step_idx]
		];
		OB_LIST << ob;
	}
	
	/**
	 * Submit an observation for a transit segment (vehicle)
	 * Records the transit details including capacity utilisation and route info
	 */
	action submit_ob_transit(float segment_duration, float dist, int ob_step_idx, float capacity) {    
		map<string,unknown> ob <- [
			"type"::"transit",
			"timestamp"::CURRENT_TIMESTAMP,
			"waiting_time"::0,
			"moving_id"::moving_id,
			"activity_id"::activity_id,
			"distance"::dist,
			"duration"::segment_duration,
			"capacity_utilization"::capacity,
			"departure_stop_name"::(ob_step_idx > 0? list_destination_stop_name[ob_step_idx-1]:""),
			"arrival_stop_name"::list_destination_stop_name[ob_step_idx],
			"by_vehicle_route_id"::list_route_id[ob_step_idx]
		];
		OB_LIST << ob;
	}
	
	/**
	 * Submit an observation for a segment travelled with a PERSONAL VEHICLE
	 * (ticket 077, lot B2). No origin stop, no arrival stop, no line: a personal
	 * vehicle has none, and pretending otherwise produces the « Unknown Unknown » that this
	 * ticket removes.
	 */
	action submit_ob_vehicle(float segment_duration, float dist, string vehicle_mode) {
		map<string,unknown> ob <- [
			"type"::"vehicle",
			"timestamp"::CURRENT_TIMESTAMP,
			"moving_id"::moving_id,
			"activity_id"::activity_id,
			"mode"::vehicle_mode,
			"distance"::dist,
			"duration"::segment_duration
		];
		OB_LIST << ob;
	}

	/**
	 * Submit an observation for the waiting time at a stop
	 * Records how long the agent waited for a vehicle
	 */
	action submit_vehicle_wait_time(float wait_duration, int ob_step_idx) {
		map<string,unknown> ob <- [
			"type"::"wait_in_stop",
			"timestamp"::CURRENT_TIMESTAMP,
			"activity_id"::activity_id,
			"duration"::wait_duration,
			"stop_name"::list_destination_stop_name[ob_step_idx-1],
			"by_vehicle_route_id"::list_route_id[ob_step_idx]
		];
		OB_LIST << ob;
	}
	
	/**
	 * Submit a final observation when the trip is finished
	 * Records the overall performance of the trip compared with the planned duration
	 */
	action submit_ob_tripfeedback(float trip_duration) {
		map<string, unknown> plan <- map<string, unknown>(raw_trip["plan"]);
		float plan_duration <- (float(plan["end_time"]) - float(plan["start_time"])) / 1000.0;
		map<string,unknown> ob <- [
			"type"::"arrival",
			"timestamp"::CURRENT_TIMESTAMP,
			"moving_id"::moving_id,
			"activity_id"::activity_id,
			"duration"::trip_duration,
			"plan_duration"::plan_duration,
			"started_at"::CURRENT_TIMESTAMP-trip_duration,
			"schedule_at"::schedule_at,
			"arrive_at"::CURRENT_TIMESTAMP,
			"expected_arrive_at"::expected_arrive_at,
			"prepare_before_seconds"::raw_trip["prepare_before_seconds"],
			"purpose"::purpose
		];
		OB_LIST << ob;
	}
	
	action submit_ob_tc_timeout(float wait_duration, int ob_step_idx) {
		map<string,unknown> ob <- [
			"type"::"tc_timeout",
			"timestamp"::CURRENT_TIMESTAMP,
			"moving_id"::moving_id,
			"activity_id"::activity_id,
			"wait_duration"::wait_duration,
			"route_id"::list_route_id[ob_step_idx],
			"stop_name"::list_destination_stop_name[ob_step_idx],
			"schedule_at"::schedule_at,
			"expected_arrive_at"::expected_arrive_at
		];
		OB_LIST << ob;
	}

	/**
	 * Get the emoji representation of the current action/state
	 * Used for the visual display of the agent's activity
	 */
	string get_action_emoji {
		if !is_idle {
			if list_route_id != nil {
				if list_route_id[step_idx] = _WALK_ {
					return PURPOSE_ICON_MAP["__WALKING__"];
				}
				else if list_route_id[step_idx] = _CAR_ {
					return PURPOSE_ICON_MAP["__DRIVING__"];
				}
				else if list_route_id[step_idx] = _BIKE_ {
					return PURPOSE_ICON_MAP["__BIKE__"];
				}
			}
			return PURPOSE_ICON_MAP["__MOVING__"];
		}
		if purpose in PURPOSE_ICON_MAP.keys {
			return PURPOSE_ICON_MAP[purpose];
		}
		return "";
	}
	
	/**
	 * Colour according to the travel state:
	 *   white  → idle (no trip)
	 *   grey   → ready/waiting before departure, or plan without movement (same location)
	 *   green  → on public transport (in the vehicle)
	 *   yellow → waiting for a PT vehicle at a stop
	 *   cyan   → on foot
	 *   red    → by car
	 *   purple → by bike / train
	 */
	rgb get_agent_color {
		if is_idle {
			return #white;
		}
		// Checked first: if the agent is physically in a PT vehicle → green guaranteed
		if on_vehicle != nil {
			return #green;
		}
		if is_ready {
			return #gray;
		}
		// main_mode="" = plan received without legs (same location): waiting on the spot → grey
		if main_mode = "" {
			return #gray;
		}
		if main_mode = _BIKE_ {
			return #purple;
		}
		if main_mode = _CAR_ {
			return #red;
		}
		if main_mode = _WALK_ {
			return #cyan;
		}

		// Others (transit waiting for a vehicle)
		return #yellow;
	}

	/**
	 * Default visual aspect for the inhabitant agents
	 * Colour according to the state (dark grey/grey/orange/red), larger size for the LLM agents
	 */
	aspect default {
		if !show_inhabitants {
			return;
		}
		rgb agent_color <- get_agent_color();
		int base_size <- is_llm_based ? 20 : 9;
		// Agents in a PT vehicle: circle (different from the usual square) slightly offset
		// to stay visible on top of the vehicle's drawing
		if on_vehicle != nil {
			draw circle(base_size * inhabitant_display_size)
				color: agent_color
				border: #white
				at: location + {base_size * inhabitant_display_size * 0.5, -base_size * inhabitant_display_size * 0.5};
		} else {
			draw square(base_size * inhabitant_display_size)
				color: agent_color
				border: true;
		}
		if show_name and is_idle = false and is_ready = false {
			draw (get_action_emoji()) at: location + {-3,1.5} anchor: #bottom_center color: agent_color font: font('Default', (is_llm_based ? 18 : 16), #bold);
			draw (person_id) at: location + {-3,1.5} anchor: #top_left color: agent_color font: font('Default', (is_llm_based ? 10 : 8), #bold);
		}
	}
}
