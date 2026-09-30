/**
* Name: MapData
* Based on the internal empty template. 
* Author: dung
* Tags: 
*/


model Settings

global {
	// feature toggle
	bool ft_public_transport_eval <- false;
	bool ft_evaluate_modality_choices <- false;
	// A/B mode with a common prefix: /sync waits for the controller's acknowledgement.
	bool prefixe_commun <- false;
	string PREFIXE_SYNC_REPONSE <- "";
	bool PREFIXE_SYNC_ERREUR <- false;
	
	string diff_arrival_time_file -> "../results/diff_arrival_time.csv";
	string evaluate_modality_choices_file -> "../results/evaluate_modality_choices.csv";
	string evaluate_density_file -> "../results/evaluate_density.csv";
	
	int LLMAGENT_QUERY_MOVE_BATCH_SIZE <- 200;
	
	// TODO: uncomment this line to enable fixed_date GTFS lookup 
//	 date GTFS_FIXED_DATE <- date([2025,3,5,0,0,0]);
	date GTFS_FIXED_DATE <- nil;
	
	// config
	date starting_date <- date([2026,3,16,5,0,0]); // Monday 5 a.m.
	
	// Global helper variables
	date UTC_START_DATE <- date([1970,1,1,0,0,0]);
	// Predictive backpressure (ticket 003): state pushed by Python via the topic
	// system/throttle. Makes it possible to display the degraded regime in the experiment UI.
	bool THROTTLE_ACTIVE <- false;      // true = simulation throttled by the predictive control
	float LLM_RATE_PER_MIN <- 0.0;      // actual LLM+OTP completion rate (tasks/min)
	float SIM_RATIO_PYTHON <- 0.0;      // simulation speed seen by Python (sim s / real s)

	int CURRENT_TIMESTAMP -> int(current_date - UTC_START_DATE);
	int SECONDS_IN_24H <- 24*3600;
	int CURRENT_TIMESTAMP_24H -> (int(current_date - UTC_START_DATE)) mod SECONDS_IN_24H;
	
	// Helper variables for display (day of the week and elapsed time)
	map<int, string> DAYS_OF_WEEK <- [1::"Lundi", 2::"Mardi", 3::"Mercredi", 4::"Jeudi", 5::"Vendredi", 6::"Samedi", 7::"Dimanche"];
	string current_day_name -> DAYS_OF_WEEK[current_date.day_of_week]; // day_of_week goes from 1 (Monday) to 7 (Sunday)
	
	int elapsed_seconds -> int(current_date - starting_date);
	int elapsed_days -> elapsed_seconds div SECONDS_IN_24H;
	int elapsed_hours -> (elapsed_seconds mod SECONDS_IN_24H) div 3600;
	int elapsed_minutes -> (elapsed_seconds mod 3600) div 60;
	
	// Formatted string ready to be displayed (e.g."Lundi | Écoulé : 2j 4h 30m")
	string display_time_info -> current_day_name + " " + current_date+ " | Écoulé : " + (elapsed_days > 0 ? string(elapsed_days) + "j " : "") + string(elapsed_hours) + "h " + string(elapsed_minutes) + "m";

	// Shape
	// World = envelope of the POLYGON of the 453 communes of the EMC² 2023 survey (ticket 031, G1),
	// exported by scripts/data/gama/export_perimetre_shapefile.py (WGS84, like routes.shp).
	// Before 2026-09-03: envelope(routes0_shape_file), the footprint of the Tisséo lines — 163 homes
	// of the population were outside it (perimeter report). This file is loaded FIRST:
	// it is the one that sets GAMA's internal projection (WGS84 → UTM, as routes.shp did before it).
	file perimetre_shape_file <- shape_file("../includes/perimetre_453.shp");
	file routes0_shape_file <- shape_file("../includes/routes.shp");
	//file shape_file_buildings <- file("../includes/building.shp");
	file stops0_shape_file <- shape_file("../includes/stops.shp");
	file trip_info_file <- json_file("../includes/trip_info.json");
	map<string, unknown> TRIP_INFO <- trip_info_file.contents;
	list<map<string, unknown>> TRIP_LIST <- TRIP_INFO["trip_list"];

	geometry shape <- envelope(perimetre_shape_file);
	// Footprint of the PT lines (Tisséo), for the coverage warning at loading.
	geometry ROUTES_ENVELOPE <- envelope(routes0_shape_file);
	
	// Readable name of each GTFS `route_type` served by the layers. Aligned on
	// `settings.gtfs.gtfs_modality_name_map` (services/llm-agents/settings.py), which is what the
	// agent's prompt reads: two different labels for the same type would make the
	// GAMA log and the prompt's incomparable.
	map<float, string> ROUTE_TYPE_NAME <- [
		0::"T1/Tram",
		1::"Metro",
		2::"Train",   // TER (SNCF Voyageurs), entered the layers on 2026-09-04
		3::"Bus",     // Tisséo urban AND liO coaches
		6::"Teleo"
	];

	// ⚠ The two tables below are indexed by the GTFS `route_type`. A MISSING
	// key raises nothing: the width becomes nil (line without thickness) and the capacity
	// becomes nil, so `is_full` is true from the first passenger — a vehicle nobody
	// can board, without a word in the log. The safeguard that lists the
	// types actually present in the layers lives in `PublicTransport.gaml`
	// (`recenser_les_route_types`): it is the real fix, the keys are only
	// today's case.
	map<float, float> ROUTE_DISPLAY_WIDTH <- [
		0::20, // T1:
		1::30, // Metro A, B
		2::25, // TER — display choice: between the tram and the metro. The 266 shapes of the
		       // TER cross the whole perimeter (86 km wide); any thinner, a regional
		       // line disappears at the scale of the world. (266 and not 34 since
		       // 2026-09-04: one shape per service pattern, cf. scripts/data/gama/gtfs_traces.py.
		       // They overlap on the same corridors, the display does not change.)
		3::3, // Bus
		6::8 // Teleo
	];

	map<float, int> VEHICLE_MAX_CAPACITY <- [
		0::200,
		1::200,
		// TER — ORDER OF MAGNITUDE, not a measurement: the GTFS publishes no capacity, and
		// a trainset is not a bus. The rolling stock actually in service in Occitanie brackets the
		// value: four-car Régiolis type M, suburban layout, single class,
		// **206 seats including 46 folding seats** (Rail Passion, « L'Occitanie commande
		// de nouveaux Régiolis »); Regio 2N, **up to 500 places including 343 seated**
		// (Région Occitanie, laregion.fr/Les-nouvelles-rames-TER). 300 sits between the
		// two families, standing places included — like the other entries of this table,
		// which count seated AND standing (100 for an articulated bus).
		2::300,
		3::100,
		6::1500
	];

	map<string, string> PURPOSE_ICON_MAP <- [
		"home"::"🏠",
		"work"::"🏢",
		"education"::"🏢",
		"shop"::"🛒",
		"leisure"::"🎵",
		"other"::"",
		"__MOVING__"::"🚌",
		"__WALKING__"::"🚶",
		"__BIKE__"::"🚲",
		"__DRIVING__"::"🚗"
	];
	
//	string POPULATION_CRS <- "EPSG:2154";
	string POPULATION_CRS <- "EPSG:4326";

	// Config persistence — reloads simulation parameters across GAMA sessions
	string SIM_CONFIG_PATH <- "../config/sim_params.yaml";
	list<string> _cfg_lines <- file_exists(SIM_CONFIG_PATH) ? list<string>(text_file(SIM_CONFIG_PATH).contents) : list<string>([]);
	string _cfg_pop <- first(_cfg_lines where (each index_of "population_size:" = 0));
	string _cfg_llm  <- first(_cfg_lines where (each index_of "part_of_llm_based_agents:" = 0));
	string _cfg_ltm  <- first(_cfg_lines where (each index_of "long_term_memory_enabled:" = 0));
	string _cfg_ltsr <- first(_cfg_lines where (each index_of "long_term_self_reflect_enabled:" = 0));
	string _cfg_days <- first(_cfg_lines where (each index_of "simulation_max_days:" = 0));
	string _cfg_prefixe <- first(_cfg_lines where (each index_of "prefixe_commun:" = 0));
	string _cfg_acc  <- first(_cfg_lines where (each index_of "accidents_enabled:" = 0));

	int population_size <- 100;
	float part_of_llm_based_agents <- 1.0;
	bool long_term_memory_enabled <- true;
	bool long_term_self_reflect_enabled <- true;
	int simulation_max_days <- 7;
	// Accidents drawn at random on the roads (ticket 070). TRUE BY DEFAULT since 2026-09-15,
	// by the author's decision: the realistic regime becomes the ordinary one, and it is its absence
	// that must be requested. ⚠ Every run therefore carries accidents — harmless as long as they
	// lengthen no trip, to be watched as soon as the suffered delay is wired in.
	bool accidents_enabled <- true;

	// Accident PLACED BY HAND (ticket 070, work item F). The figures come from this
	// placement: the random draw, at real frequency, only hits ~0.6 trips per
	// simulated day with 1,000 agents. Defaults set on the Toulouse ring road at Empalot,
	// morning peak hour.
	float accident_pose_lat <- 43.5735;
	float accident_pose_lon <- 1.4330;
	int accident_pose_heure <- 8;       // wall-clock hour of the current simulated day
	int accident_pose_duree <- 45;      // minutes

	// Number of cycles (loop iterations) for the prompt calibration,
	// launched on demand from the GUI (button"Lancer la calibration du prompt").
	int calibration_cycles <- 20;

	action save_sim_config {
		write "Save config";
		if (population_size>0){
			string content <- "population_size: " + string(population_size) + "\n"
				+ "part_of_llm_based_agents: " + string(part_of_llm_based_agents) + "\n"
				+ "long_term_memory_enabled: " + string(long_term_memory_enabled) + "\n"
				+ "long_term_self_reflect_enabled: " + string(long_term_self_reflect_enabled) + "\n"
				+ "simulation_max_days: " + string(simulation_max_days) + "\n"
				+ "prefixe_commun: " + string(prefixe_commun) + "\n"
				+ "accidents_enabled: " + string(accidents_enabled);
			save content to: SIM_CONFIG_PATH format: "text" rewrite: true;
		}
	}

	action load_sim_config {
		// Simulation scenario parameters — sent to the Python controller at /init
		write "Load config";
		population_size <- (_cfg_pop != nil) ? int((_cfg_pop split_with ":")[1]) : 0;
		part_of_llm_based_agents <- (_cfg_llm != nil) ? float(string((_cfg_llm split_with ":")[1]) replace(" ", "")) : 1.0;
		long_term_memory_enabled <- (_cfg_ltm != nil) ? ((_cfg_ltm split_with ":")[1] contains "true") : false;
		long_term_self_reflect_enabled <- (_cfg_ltsr != nil) ? (string((_cfg_ltsr split_with ":")[1]) contains "true") : false;
		simulation_max_days <- (_cfg_days != nil) ? int(string((_cfg_days split_with ":")[1]) replace(" ", "")) : 7;
		prefixe_commun <- (_cfg_prefixe != nil) ? ((_cfg_prefixe split_with ":")[1] contains "true") : false;
		// Absent from the file = the regime was never decided for this machine: the default is taken,
		// now TRUE. A `sim_params.yaml` older than 2026-09-15 does not carry the
		// key and will therefore enable accidents — this is intended, and it is why the tickets that
		// launch a run carry a warning about this parameter.
		accidents_enabled <- (_cfg_acc != nil) ? ((_cfg_acc split_with ":")[1] contains "true") : true;
	}

	reflex auto_save_sim_config when: cycle = 2 {
		do save_sim_config;
	}
	
	init {
		do load_sim_config;
		// Coverage of the world by the PT network (ticket 031, G1). Expected since the switch to the
		// polygon of the 453 communes: Tisséo does not serve the 3rd ring, its lines do not cover
		// the world — the warning says by how much, so that it is not discovered agent by agent.
		if (!(ROUTES_ENVELOPE covers shape)) {
			float part_couverte <- shape.area > 0 ? (shape inter ROUTES_ENVELOPE).area / shape.area : 0.0;
			write "[PERIMETRE] World = envelope of the perimeter of the 453 communes (" + int(shape.width / 1000) + " x " + int(shape.height / 1000) + " km); the PT lines (routes.shp) only cover " + int(part_couverte * 100) + "% of it — agents outside their footprint have no Tisséo public transport.";
		} else {
			write "[PERIMETRE] World = envelope of the perimeter of the 453 communes, fully covered by the PT lines.";
		}
	}

}
