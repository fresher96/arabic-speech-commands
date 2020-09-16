
class ClassDict:

    dct = ['cancel',        'one',          'move',         'record',       'stop',
           'down',          'backward',     'zoom out',     'close',        'eight',
           'undo',          'options',      'previous',     'open',         'rotate',
           'next',          'disable',      'two',          'six',          'start',
           'ok',            'right',        'zoom in',      'yes',          'up',
           'left',          'four',         'digit',        'seven',        'enter',
           'enable',        'nine',         'receive',      'direction',    'five',
           'forward',       'zero',         'three',        'no',           'send']

    @staticmethod
    def getId(name):
        return ClassDict.dct.index(name)

    @staticmethod
    def getName(idx):
        return ClassDict.dct[idx] if idx < ClassDict.len() else 'silence'

    @staticmethod
    def len():
        return len(ClassDict.dct)
