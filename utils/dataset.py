import torch


class Dataset:
    """
    A dict to store collected data,
    init:
        data = Dataset(batch_size, num_envs)
        # create a buffer with shape (batch_size, num_envs, *single_buffer_shape) for key 'obs'
        data.AddBuffer('obs', single_buffer_shape)
    record data:
        # obs_buffer.shape is (num_envs, *single_buffer_shape)
        # data['obs'][batch_idx, :] = obs_buffer
        data.Record('obs', batch_idx, obs_buffer)
    acquire data:
        batched_obs = data['obs']
    iterate:
        for k in data.keys():
            print(data[k].shape) # do something you want
    """

    def __init__(self, batch_size, num_envs):
        self.head_shape = (batch_size, num_envs)
        self.data = dict()

    def AddBuffer(self, buf_name, single_buf_shape, dtype=None, device=None):
        self.data[buf_name] = torch.zeros(*self.head_shape, *single_buf_shape, dtype=dtype, device=device)

    def Record(self, buf_name, idx, buf):
        named_data = self.data.get(buf_name, None)
        if named_data is None:
            raise KeyError(buf_name)
        elif buf.shape != named_data.shape[1:]:
            raise ValueError("expected buffer shape: " + str(named_data.shape[1:]) + ", got: " + str(buf.shape))
        named_data[idx, :] = buf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, buf_name):
        return self.data[buf_name]

    def keys(self):
        return self.data.keys()
