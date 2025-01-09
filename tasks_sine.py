import numpy as np
import torch
import scipy.stats as stats


class RegressionTasksSinusoidal:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self):
        self.num_inputs = 1
        self.num_outputs = 1
        
        self.amplitude_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]

        self.input_range = [-5, 5]
        self.num_classes = 1


    def sample_tasks(self, num_tasks, num_points, init_dis):
        # sample params from uniform dist to generate tasks
        if init_dis == 'Uniform':
            amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_tasks)
            phase = np.random.uniform(self.phase_range[0], self.phase_range[1], num_tasks)
        elif init_dis == 'Normal':
            amplitude = np.random.normal(2.5, 2, num_tasks)
            phase = np.random.normal(1.5, 1, num_tasks)
            amplitude = np.clip(amplitude, self.amplitude_range[0], self.amplitude_range[1])
            phase = np.clip(phase, self.phase_range[0], self.phase_range[1])
        
        x, y = self.sample_datapoints(num_tasks, num_points, amplitude, phase)
        return x, y

    
    def sample_init_param(self, num_tasks, init_dis):
        # sample batch of hyper-params of tasks for the initialization of tasks
        if init_dis == 'Uniform':
            amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_tasks)
            phase = np.random.uniform(self.phase_range[0], self.phase_range[1], num_tasks)
        elif init_dis == 'Normal':
            amplitude = np.random.normal(2.5, 2, num_tasks)
            phase = np.random.normal(1.5, 1, num_tasks)
            amplitude = np.clip(amplitude, self.amplitude_range[0], self.amplitude_range[1])
            phase = np.clip(phase, self.phase_range[0], self.phase_range[1])
        
            
        init_param_tensor = torch.zeros(num_tasks, 2)
        
        for i in range(num_tasks):
            init_param_tensor[i, 0] = amplitude[i]
            init_param_tensor[i, 1] = phase[i]
        
        return init_param_tensor
        
        
    def generate_tasks(self, batch_size, num_points, trasformed_param):
        # generate tasks by configuring the transfermed param.
        amplitudes = trasformed_param[:, 0]
        phases = trasformed_param[:, 1]
        x, y = self.sample_datapoints(batch_size, num_points, amplitudes, phases)
        return x, y
       
        
    def sample_datapoints(self, batch_size, num_points, amplitudes, phases):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """

        inputs = torch.rand((batch_size, num_points, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        inputs = inputs.cuda()
        
        outputs = torch.rand((batch_size, num_points, self.num_inputs))
        for i in range(batch_size):
            outputs[i] = torch.sin(inputs[i] - phases[i]) * amplitudes[i]

        return inputs, outputs
    
    
    def sample_meta_dataset(self, batch_x, batch_y, update_batch_size):
        inputa = batch_x[:, :self.num_classes*update_batch_size, :]
        labela = batch_y[:, :self.num_classes*update_batch_size, :]
        inputb = batch_x[:, self.num_classes*update_batch_size:, :] # b used for testing
        labelb = batch_y[:, self.num_classes*update_batch_size:, :]
        return inputa, labela, inputb, labelb
    
    def sample_noise_meta_dataset(self, batch_x, batch_y, update_batch_size):
        inputa = batch_x[:, :self.num_classes*update_batch_size, :]
        labela = batch_y[:, :self.num_classes*update_batch_size, :]
        inputb = batch_x[:, self.num_classes*update_batch_size:, :] # b used for testing
        labelb = batch_y[:, self.num_classes*update_batch_size:, :]

        noise_scale = 0.1  # Adjust noise scale as needed
        noisea = torch.Tensor(np.random.normal(scale=noise_scale, size=labela.shape))
        labela += noisea
        return inputa, labela, inputb, labelb
