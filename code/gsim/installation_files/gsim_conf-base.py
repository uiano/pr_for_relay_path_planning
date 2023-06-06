""" 
This file contains configuration specific to each user. 

It is intended to be ignored in .gitignore.

If you wish to distribute your repository, you may consider editing a copy of
this file and placing it in the root folder of your repository with name
.gsim_conf_default.py. That file will be used instead of the present file to
create gsim-conv.py when running install.sh. This is useful e.g. to set the
value of the variable `module_name` below. 

More information in gsim/install.sh

"""

# Select an experiment file:
module_name = "experiments.example_experiments"

# GFigure
import gsim.gfigure

gsim.gfigure.title_to_caption = False
gsim.gfigure.default_figsize = (5.5, 3.5)  # `None` to let plt choose.

#log.setLevel(logging.DEBUG)
import logging.config

cfg = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '{levelname}:{name}:{module}: {message}',
            'style': '{',
        },
        'standard': {
            'format': '{levelname}:{asctime}:{name}:{module}: {message}',
            'style': '{',
        },
        'verbose': {
            'format':
            '{levelname}:{asctime}:{name}:{module}:{process:d}:{thread:d}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        # 'file': {
        #     'level': 'INFO',
        #     'class': 'logging.FileHandler',
        #     'filename': os.path.join(BASE_DIR, LOGGING_DIR, 'all.log'),
        #     'formatter': 'standard'
        # },
        'console': {  # This one is overridden in settings_server.py
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'experiments': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': True,
        },
    }
}
logging.config.dictConfig(cfg)