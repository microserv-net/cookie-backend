"""Cookie's backend: the mind behind the voice.

The frontend (`cookie-interface`) owns the microphone, the speaker and the
orb. Everything here is about deciding what to say and what to do — and,
because the first machine this runs on holds roughly one large model in
memory, about being interruptible while doing it.
"""

__version__ = "0.1.0"
PROTOCOL = "cookie-interface/1"
