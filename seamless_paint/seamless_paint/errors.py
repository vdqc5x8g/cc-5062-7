"""Shared exception type for expected invalid painting input."""


class SeamlessInputError(ValueError):
    """Invalid painting parameters (bad radius, policy, lattice, ...).

    Raised for expected client-side input problems so the web layer can report
    them back to the painting window without logging a server stack trace.
    """
