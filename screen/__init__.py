"""Mechanism screening: measure signal against noise before building a search.

Every module here exists to answer one question about a *candidate* mechanism --
does its genome move the loss by more than seed noise does? -- and to answer it
in GPU-hours rather than GPU-weeks.

The programme this implements is in the plan; the short version is that two
previous systems in this repository (SOMA, GNOMON) were specified, implemented
and tested at length around mechanisms that were later measured to contribute
nothing. `docs/gnomon/00-postmortem.md` §0.3 names the process failure: "A
conformance suite cannot detect that the thing being conformed to does not
matter." So nothing here is specified in detail until a screen says there is
something worth specifying.
"""
