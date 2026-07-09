import sys
import time
import importlib

import numpy as np

import KratosMultiphysics
import KratosMultiphysics.FluidDynamicsApplication as FDApp

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters, gt_solver):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model,project_parameters, flush_frequency=10.0):
            super().__init__(model,project_parameters)

            # Add extra nodal variables
            self.vel_grad_x_var = FDApp.ADJOINT_FLUID_VECTOR_1
            self.vel_grad_y_var = FDApp.ADJOINT_FLUID_VECTOR_2
            self.vel_grad_z_var = FDApp.ADJOINT_FLUID_VECTOR_3
            main_model_part = self._GetSolver().main_model_part
            main_model_part.AddNodalSolutionStepVariable(self.vel_grad_x_var)
            main_model_part.AddNodalSolutionStepVariable(self.vel_grad_y_var)
            main_model_part.AddNodalSolutionStepVariable(self.vel_grad_z_var)

            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()

            self.computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()

            for node in self.computing_model_part.Nodes:
                x = node.X0
                y = node.Y0
                viscosity_val = gt_solver.compute_viscosity_at_points(x,y)
                
                node.SetSolutionStepValue(KratosMultiphysics.DYNAMIC_VISCOSITY, viscosity_val)

            # Keep only if initializing values to ground truth
            # for node in self.computing_model_part.Nodes:
            #     x = node.X0
            #     y = node.Y0

                # u_x = 0
                # u_y = 0
                # p = 0
                # u_x, u_y, p = gt_solver.compute_ground_truth_at_points(x,y)

                # node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_X, u_x)
                # node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_Y, u_y)
                # node.SetSolutionStepValue(KratosMultiphysics.PRESSURE, p)

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()

            strategy = self._GetSolver()._GetSolutionStrategy()
            print(strategy.Info())

            self.computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

            self.print_relative_errors()

        def print_relative_errors(self):
            self.compute_gradients_2D()

            sum_sq_error_p = 0.0
            sum_sq_gt_p = 0.0
            
            sum_sq_error_grad_u = 0.0
            sum_sq_gt_grad_u = 0.0

            for node in self.computing_model_part.Nodes:
                # Get Node coordinates
                x, y = node.X, node.Y
                
                # Get Ground Truth analytical values at this point/time
                gt_p, gt_grad_u = gt_solver.compute_p_and_grad_u_at_point(x,y)
                # print(f"Node ID: {node.Id}, Coordinates: ({x}, {y}), Ground Truth p: {gt_p}, Ground Truth Grad(u):\n{gt_grad_u}")
                
                # Get Kratos simulated values
                sim_p = node.GetSolutionStepValue(KratosMultiphysics.PRESSURE)
                sim_grad_u = self.get_velocity_gradient_matrix_2D(node)
                
                # Pressure Error Calculation ($L_2$ norm contribution)
                sum_sq_error_p += (sim_p - gt_p) ** 2
                sum_sq_gt_p += gt_p ** 2
                
                # Velocity Gradient Error Calculation (Frobenius norm contribution)
                # sim_grad_v is typically a Matrix or flattened array in Kratos
                grad_error = sim_grad_u - gt_grad_u
                sum_sq_error_grad_u += np.sum(grad_error ** 2)
                sum_sq_gt_grad_u += np.sum(gt_grad_u ** 2)
                
            # Finalize Relative Errors
            rel_error_p = np.sqrt(sum_sq_error_p) / (np.sqrt(sum_sq_gt_p) + 1e-12)
            rel_error_grad_u = np.sqrt(sum_sq_error_grad_u) / (np.sqrt(sum_sq_gt_grad_u) + 1e-12)
                
            print(f"[METRICS] Rel Error p: {rel_error_p:.6f} | Rel Error Grad(u): {rel_error_grad_u:.6f}")

        def get_velocity_gradient_matrix_2D(self, node):
            gradient_matrix = np.zeros((2, 2))
            gradient_matrix[0, :] = np.array(node.GetSolutionStepValue(self.vel_grad_x_var))[:2]
            gradient_matrix[1, :] = np.array(node.GetSolutionStepValue(self.vel_grad_y_var))[:2]
            return gradient_matrix.T

        def compute_gradients_2D(self):

            var_combinations = [
                (KratosMultiphysics.VELOCITY_X, self.vel_grad_x_var),
                (KratosMultiphysics.VELOCITY_Y, self.vel_grad_y_var),
            ]
            
            for var, grad_var in var_combinations:
                local_grad_process = KratosMultiphysics.ComputeNodalGradientProcess2D(
                    self.computing_model_part, 
                    var,
                    grad_var # Destination variable
                )
                local_grad_process.Execute()

    return AnalysisStageWithFlush(global_model, parameters)

def run_simulation(gt_solver):
    project_parameters_path = "kratos_simulation_tests/first_order_stokes_variable_viscosity_bvs_gl/kratos_files/ProjectParameters.json"
    with open(project_parameters_path, 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters, gt_solver)
    simulation.Run()
