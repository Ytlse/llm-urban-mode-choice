/**
* Name: CityTransport
* Based on the internal empty template.
* Author: dung
* Tags:
*
* Description: Multi-agent simulation model for urban transportation in Toulouse.
* This model simulates public transport systems with LLM-powered agents making
* transportation decisions in a realistic urban environment.
*/


model City

// Import model components
//import "Density.gaml"  // Commented out - density-based modeling

// Core settings and configuration
import "Settings.gaml"

//import "OSMFileImport.gaml"  // Commented out - OSM data import

// Public transportation system
import "PublicTransport.gaml"

// Human agents (inhabitants)
import "Inhabitant.gaml"

// LLM-powered intelligent agents
import "LLMAgent.gaml"

/* Insert your model definition here */

global {
    // GTFS Calendar date range for the simulation
    date _gtfs_calendar_start_date;
    date _gtfs_calendar_end_date;


    init {
        // Load calendar information from GTFS data
        map calendar_info <- TRIP_INFO["calendar"] as map;
        list<string> t_dates <- calendar_info["dates"] as list;
        // TEXT mask since 2026-09-15 (ticket 075): see PublicTransport.gaml.
        map<string, string> t_data <- calendar_info["data"] as map;

        // Create the travel agent factory with GTFS data
        create travel_agent_factory number: 1 with: [
            data_trip_list:: TRIP_LIST,
            trip_dates_list:: t_dates,
            trip_calendar_map:: t_data
        ];

        // Initialize GTFS calendar date range
        list<string> _dates_str <- calendar_info["dates"];
        _gtfs_calendar_start_date <- date(_dates_str[0]);
        _gtfs_calendar_end_date <- date(_dates_str[length(_dates_str)-1]);
    }

    // Actual stop at the requested horizon (ticket 008, A5). The statement was
    // commented out: the reflex wrote a message and the simulation went on, so
    // that `simulation_max_days` stopped nothing and a 24 h run produced three.
    //
    // This reflex lives in `global` and not in `experiment`: `pause` is an action of
    // the SIMULATION agent, the experiment agent does not expose it (neither `pause` nor `halt` —
    // the model does not compile if it is placed there). It is also the form used by the
    // models shipped with GAMA.
    //
    // `pause` and not `die`: the clock stops without killing the agents, the outputs
    // stay inspectable, and the Python controller is not interrupted — its
    // ongoing writes (moves.csv, llm_exchanges.jsonl) complete normally.
    // A `die` would cut them off abruptly.
    reflex stop_after_max_days when: simulation_max_days > 0 and int(current_date - starting_date) >= simulation_max_days * 86400 {
        write "Simulation stopped after " + simulation_max_days + " simulated day(s).";
        do pause;
    }

    reflex stop_on_prefix_sync_error when: prefixe_commun and PREFIXE_SYNC_ERREUR {
        write "[ALARME] Simulation stopped: Préfixe commun interrompu : /sync acknowledgement missing or invalid.";
        do pause;
    }

}

// Species definitions

/**
 * Activity location species - represents points of interest in the city
 * Used for visualizing activity locations on the map
 */
species activity_loc {
    aspect default {
        draw circle(100) color: #green;
    }
}

/**
 * Main simulation experiment with GUI interface
 * Defines simulation parameters, display settings, and data collection
 */
experiment e type: gui {
    // Simulation time step (2 minutes)
    float step <- 120 #s;

    // GTFS visualization parameters
    parameter "Vehicle Size" category:"GTFS" var: vehicle_display_size <- 20.0 among: [5.0, 10.0, 20.0, 30.0, 40.0];
    parameter "Always show GTFS Routes" category:"GTFS" var: show_always_show_gtfs_routes <- true;
    parameter "Show TRAM routes" category:"GTFS" var: show_type_tram <- false;
    parameter "Show METRO routes" category:"GTFS" var: show_type_metro <- false;
    parameter "Show RAIL routes" category:"GTFS" var: show_type_rail <- false;
    parameter "Show BUS routes" category:"GTFS" var: show_type_bus <- false;
    parameter "Show TELEO routes" category:"GTFS" var: show_type_teleo <- false;
    parameter "Show Label density" category:"GTFS" var: show_label_density <- 0.5 among: [0.5, 1, 5, 10, 25, 50, 100];

    // Inhabitant visualization parameters
    parameter "Inhabitant Size" category:"Inhabitants" var: inhabitant_display_size <- 10.0 among: [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0];
    parameter "Show Inhabitants" category: "Inhabitants" var: show_inhabitants <- true;
    parameter "Show Inhabitant Label density" category: "Inhabitants" var: show_inhabitants_label_density <- 100 among: [0, 5, 10, 25, 50, 100];

    // Verbose output controls
    parameter "Public Transport" category:"Verbose" var: pt_verbose <- false;

    // Network — overridden by the headless launcher (make run OFFLINE=1) to
    // point to the Docker controller (http://controller:8002). The defaults
    // (localhost, LLMAgent.gaml) serve the GUI mode. Not persisted in
    // sim_params.yaml: the injection at `load` therefore stays effective.
    parameter "Controller HTTP URL" category: "Network" var: http_url;
    parameter "Controller HTTP port" category: "Network" var: http_port;

    // Simulation scenario parameters — sent to the Python controller at /init
    // Values are persisted to/from GAMA/CityTransport/config/sim_params.yaml
    parameter "Population size" category: "Simulation" var: population_size min: 0 max: 1000 step: 100;
    parameter "Part of LLM-based agents" category: "Simulation" var: part_of_llm_based_agents <-1.0 min: 0.0 max: 1.0;
    parameter "Long-term memory" category: "Simulation" var: long_term_memory_enabled <- true;
    parameter "Long-term self-reflection" category: "Simulation" var: long_term_self_reflect_enabled <- true;
    parameter "Max simulated days (0 = unlimited)" category: "Simulation" var: simulation_max_days min: 0 max: 365 step: 1;
    parameter "Préfixe commun A/B" category: "Simulation" var: prefixe_commun;
    // Ticket 070 — accidents EXIST (drawn from the BAAC law, placed on an edge,
    // logged). No trip duration is modified at this stage.
    // TRUE BY DEFAULT since 2026-09-15: unticking is now the explicit gesture.
    parameter "Accidents sur les axes" category: "Simulation" var: accidents_enabled <- true;

    // Evaluation features
    parameter "Public Transport - Dump stop arrival diff time" category:"Features" var: ft_public_transport_eval <- false;
    parameter "Evaluate - Multimodal Choices" category:"Evaluation" var: ft_evaluate_modality_choices <- true;

    // Accident placed by hand (ticket 070, work item F) — button below.
    parameter "Accident posé - latitude" category: "Accidents" var: accident_pose_lat;
    parameter "Accident posé - longitude" category: "Accidents" var: accident_pose_lon;
    parameter "Accident posé - heure (0-23)" category: "Accidents" var: accident_pose_heure min: 0 max: 23 step: 1;
    parameter "Accident posé - durée (min)" category: "Accidents" var: accident_pose_duree min: 1 max: 600 step: 5;

    // Places a CHOSEN accident on the edge closest to the point, today at the given hour.
    // Same mechanism and same world as the randomly drawn accidents: only the trigger
    // differs. ⚠ No effect if « Accidents sur les axes » is unticked — the controller refuses
    // the placement and says so in its log.
    user_command "Poser un accident maintenant" category: "Accidents" {
        ask simulation {
            ask llm_agent_sync {
                do poser_accident;
            }
        }
    }

    // Prompt calibration — launched on demand via the button below
    parameter "Calibration - cycles (itérations)" category: "Calibration" var: calibration_cycles min: 1 max: 200 step: 1;

    // GUI button: triggers a background prompt-calibration run in the controller
    // (POST /calibrate) with the configured number of cycles.
    user_command "Lancer la calibration du prompt" category: "Calibration" {
        ask simulation {
            ask llm_agent_sync {
                do launch_calibration;
            }
        }
    }


    // Building
    //parameter "Shapefile for the buildings:" var: shape_file_buildings category: "GIS";

    // The stop at the `simulation_max_days` horizon is declared in the `global` block
    // (ticket 008, A5): `pause` is an action of the simulation agent, not of the experiment
    // agent.

    // Save arrival time metrics every 10 minutes
    reflex save_csv when: ft_public_transport_eval and every(10#mn) {
        float max_early <- max(public_vehicle collect each.metrics_diff_arrival_time_positive);
        float max_late <- min(public_vehicle collect each.metrics_diff_arrival_time_negative);
        float mean_early <- mean(public_vehicle collect each.metrics_diff_arrival_time_positive);
        float mean_late <- mean(public_vehicle collect each.metrics_diff_arrival_time_negative);
        int count_metro <- length(public_vehicle where (each.route_type = TYPE_METRO));
        save [time,max_early,max_late,mean_early,mean_late,count_metro] to: diff_arrival_time_file
                    format:"csv" rewrite: time <= 10#mn;
    }

    // Save modality choice data every minute
    reflex save_trip_csv when: ft_evaluate_modality_choices and every(1#mn) {
        if time <= 1#mn {
            string person_id <- "";
            int route_type <- 0;
            string moving_id <- "";
            save [gama.machine_time,CURRENT_TIMESTAMP,person_id,route_type,moving_id] to: evaluate_modality_choices_file
                    format:"csv" rewrite: true;
        }
        // Record active trips with transport mode
        ask inhabitant {
            if self.on_vehicle != nil and !dead(self.on_vehicle) and self.target_location != nil {
                save [gama.machine_time,CURRENT_TIMESTAMP,person_id,self.on_vehicle.route_type,self.moving_id] to: evaluate_modality_choices_file
                    format:"csv" rewrite: false;
            }
        }
    }

    // Save inhabitant location data every 5 minutes
    reflex save_loc_csv when: every(5#mn) {
        if time <= 5#mn {
            float lon <- 0.0;
            float lat <- 0.0;
            string trip_id <- nil;
            string person_id <- nil;
            save [gama.machine_time,CURRENT_TIMESTAMP,person_id,trip_id,lon,lat] to: evaluate_density_file
                    format:"csv" rewrite: true;
        }

        ask inhabitant where (!each.is_idle) {
            string trip_id <- self.on_vehicle != nil ? self.on_vehicle.trip_id : nil;
            point ploc <- point(location CRS_transform(POPULATION_CRS));
            save [gama.machine_time,CURRENT_TIMESTAMP,person_id,trip_id, ploc.x, ploc.y]
                to: evaluate_density_file
                format:"csv" rewrite: false;
        }
    }
   

    
    // Output displays and charts
    output {
    	
    
        monitor "NB agents active" value: length(inhabitant where each.is_active);
        
        // Main map display showing all simulation elements
        display map {
            // Real-time information overlay
            graphics Strings {
                // Current simulation date
                draw "Date: " + string(display_time_info)
                    at: {10, -2000} 
                    anchor: #top_left
                    border: #black font: font("Geneva", 20)
                    wireframe: true width: 2;
                // GTFS calendar date range
                draw "GTFS Date: " + string(_gtfs_calendar_start_date, "MM/dd") + " - " + string(_gtfs_calendar_end_date, "MM/dd")
                    at: {-10000, 1200} 
                    anchor: #top_left
                    border: #black font: font("Geneva", 12)
                    wireframe: true width: 1;
                // Ready agents counter
                draw "Ready Agents: " + string(length(inhabitant where (each.is_ready))) + " / " + string(length(inhabitant))
                    at: {-10000, 2400} 
                    anchor: #top_left
                    border: #black font: font("Geneva", 12)
                    wireframe: true width: 1;
                // Active agents counter
                draw "Active Agents: " + string(length(inhabitant where (each.is_active))) + " / " + string(length(inhabitant))
                    at: {-10000, 3600} 
                    anchor: #top_left
                    border: #black font: font("Geneva", 12)
                    wireframe: true width: 1;
                // Total activities completed
                draw "Total Activities: " + string(sum(inhabitant collect (each.total_activities)))
                    at: {-10000, 4800} 
                    anchor: #top_left
                    border: #black font: font("Yellow", 12)
                    wireframe: true width: 1;
                    
                draw "Inactif "
                    at: {200, 39400} 
                    anchor: #top_left
                    border: #darkgray font: font("Geneva", 12, #bold)
                    wireframe: true width: 2;
                
                draw "Ready for next activity "
                    at: {2500, 39400} 
                    anchor: #top_left
                    border: #gray font: font("Geneva", 12, #bold)
                    wireframe: true width: 2;
                    
                draw "Transport en commun "
                    at: {9000, 39400} 
                    anchor: #top_left
                    border: #green font: font("Geneva", 12, #bold)
                    wireframe: true width: 2;
                    
               draw "Walking "
                    at: {15500, 39400} 
                    anchor: #top_left
                    border: #orange font: font("Geneva", 12, #bold)
                    wireframe: true width: 2;
              
               draw "Car "
                    at: {18000, 39400} 
                    anchor: #top_left
                    border: #red font: font("Geneva", 12, #bold)
                    wireframe: true width: 2;

                draw "Bike "
                    at: {20000, 39400}
                    anchor: #top_left
                    border: #purple font: font("Geneva", 12, #bold)
                    wireframe: true width: 2;

                // Train (TER) — purple, like the bike: this is the official palette of the
                // repository (.claude/CLAUDE.md, « Reference: Transport Mode Colors »), which
                // gives the same colour to both. The `route_type=2` vehicles are drawn
                // with this colour (cf. PublicTransport.gaml, public_vehicle aspect).
                draw "Train "
                    at: {22000, 39400}
                    anchor: #top_left
                    border: #purple font: font("Geneva", 12, #bold)
                    wireframe: true width: 2;

            }

            //image "../includes/toulouse_map.png";

            // Species to display on map
            // species building aspect: base ; // Buildings 
            species route;        // Transport routes
            species stop;         // Transit stops
            species travel_agent_factory;  // GTFS data manager
            species inhabitant;   // Human agents
            species public_vehicle;  // Transit vehicles
            species activity_loc;    // Activity locations
        }
	
// Uncomment this to view the arrival time metrics
//		display monitor {
//			chart "Mean arrival time diff" type: series
//			{
//				data "Max Early" value: max(public_vehicle collect each.metrics_diff_arrival_time_positive) color: # green marker_shape: marker_empty style: spline;
//				data "Max Late" value: min(public_vehicle collect each.metrics_diff_arrival_time_negative) color: # red marker_shape: marker_empty style: spline;
//			}
//		}

//		display monitor refresh: every(50 #cycles) {
//			chart "Total bus" type: series
//			{
//				data "Bus" value: length(public_vehicle select (each.route_type = 3)) color: # green marker_shape: marker_empty style: spline;
//			}
//		}
	} 
}
