"""Monkey patches engine classes to be parameterizable where needed."""


def parametrizable_engines(engine):
    if engine == 'lpg-anytime':
        from up_lpg.lpg_planner import LPGAnytimeEngine
        original_init = LPGAnytimeEngine.__init__

        def new_lpg_anytime_init(self, **kwargs):
            original_init(self)
            self.parameter = []
            for param, val in kwargs.items():
                self.parameter.append(str(param))
                if val != '':
                    self.parameter.append(str(val))

        LPGAnytimeEngine.__init__ = new_lpg_anytime_init
