import json
import shutil
from pathlib import Path

from kratos_simulation_tests.first_order_stokes_variable_viscosity_bvs_gl.MainKratos import run_simulation as kratos_run_simulation
from kratos_simulation_tests.first_order_stokes_variable_viscosity_bvs_gl.ground_truth import GroundTruthSolver

def run_kratos(kratos_results_filename, kratos_node_locations_filename, gt_solver):
    kratos_run_simulation(kratos_results_filename, kratos_node_locations_filename, gt_solver)

def run_evaluation():

    elements_num = 20480

    kratos_files_path = Path("kratos_files")
    src_mdpa_file_path = kratos_files_path / f"channel_{elements_num}.mdpa"
    dst_mdpa_file_path = kratos_files_path / "channel.mdpa"

    try:
        shutil.copy2(src_mdpa_file_path, dst_mdpa_file_path)
    except:
        raise f"Could not copy {src_mdpa_file_path} to {dst_mdpa_file_path}"

    gt_solver = GroundTruthSolver(dst_mdpa_file_path)

    kratos_results_filename = "kratos_results.npy"
    kratos_node_locations_filename = "kratos_node_locations.npy"

    print("Domain bounds:", gt_solver.bounds)
    print("Ground truth analytical functions:", gt_solver.get_dirichlet_boundary_analytical_functions())

    # Apply correct delta parameter within Kratos' Project Parameters
    delta_param = gt_solver.compute_delta_parameter()
    sigma_param = gt_solver.get_sigma_parameter()
    print("Setting delta parameter to:", delta_param)
    print("Setting sigma parameter to:", sigma_param)
    project_parameters_path = kratos_files_path / "ProjectParameters.json"
    with open(project_parameters_path, "r") as f:
        kratos_project_params = json.load(f)

    formulation_config = kratos_project_params["solver_settings"]["formulation"]
    formulation_config["delta_parameter"] = delta_param
    formulation_config["sigma_parameter"] = sigma_param

    u_x_function, _, p_function = gt_solver.get_dirichlet_boundary_analytical_functions()
    boundary_params = kratos_project_params["processes"]["boundary_conditions_process_list"]
    for i in range(len(boundary_params)):
        if boundary_params[i]["python_module"] == "apply_inlet_process":
            boundary_params[i]["Parameters"]["modulus"] = u_x_function
        elif boundary_params[i]["python_module"] == "apply_outlet_process":
            boundary_params[i]["Parameters"]["value"] = p_function

    with open(project_parameters_path, "w") as f:
        json.dump(kratos_project_params, f, indent=4)

    run_kratos(kratos_results_filename, kratos_node_locations_filename, gt_solver)
