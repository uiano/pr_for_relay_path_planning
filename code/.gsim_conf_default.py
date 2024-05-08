""" gsim_conf.py is intended to be ignored in .gitignore. It is a configuration
file specific to each user.

gsim_conf-base.py can be used to create a template for gsim_conf.py specific to
each project. """

# MAYAVI
#
# To install mayavi on Mac M1:
# 1. Install Qt5 with homebrew.
# 2. export PATH="/opt/homebrew/opt/qt5/bin:$PATH"
# 3. pip -v install pyqt5 hangs at the license --> https://stackoverflow.com/questions/66546886/pip-install-stuck-on-preparing-wheel-metadata-when-trying-to-install-pyqt5
#
# With animations, to prevent mayavi from showing a dialog prompting for the delay, just comment line 69 :
#
#  BaseDialog.display_ui(ui, parent, style)
#
# in site-packages/traitsui/qt4/ui_live.py
#
use_mayavi = False

# Select an experiment file:
module_name = "experiments.paper_experiments"

# GFigure
import gsim.gfigure

gsim.gfigure.title_to_caption = True
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
