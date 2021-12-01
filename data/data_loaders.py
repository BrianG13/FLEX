from base.base_data_loader import BaseDataLoader
from data import multi_view_h36_dataset

NUMBER_WORK = 8


class h36m_loader(BaseDataLoader):
    def __init__(self, config, is_training=False, eval_mode=False):
        self.dataset = multi_view_h36_dataset.multi_view_h36_dataset(config, is_train=is_training,
                                                                     num_of_views=config.arch.n_views,
                                                                     eval_mode=eval_mode)

        batch_size = config.trainer.batch_size if is_training else 1
        super(h36m_loader, self).__init__(self.dataset, batch_size=batch_size, shuffle=is_training, pin_memory=True,
                                          num_workers=NUMBER_WORK, drop_last=True)
