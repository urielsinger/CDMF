from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from CDMF.dataset import CDMFDataModule
from CDMF.model import CDMFModule
from CDMF.config import parser

hparams = parser.parse_args()

pl.trainer.seed_everything(hparams.seed)

logger = WandbLogger(name=hparams.dataset,
                     version=datetime.now().strftime('%y%m%d_%H%M%S.%f'),
                     project='CDMF',
                     config=hparams)

datamodule = CDMFDataModule(dataset=hparams.dataset,
                            max_seq_len=hparams.max_seq_len,
                            batch_size=hparams.batch_size,
                            num_workers=hparams.num_workers)
datamodule.prepare_data()  # called only because n_items, n_users and n_features are initialized in `prepare_data()`
model = CDMFModule(n_items=datamodule.n_items,
                   n_users=datamodule.n_users,
                   n_features=datamodule.n_features,
                   emb_dim=hparams.emb_dim,
                   tau=hparams.tau,
                   per_user=hparams.per_user,
                   learning_rate=hparams.learning_rate,
                   weight_decay=hparams.weight_decay)

trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger, log_every_n_steps=1)
trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule)
