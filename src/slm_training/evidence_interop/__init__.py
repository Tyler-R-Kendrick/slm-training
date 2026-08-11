"""Thin provenance / attestation interoperability profile (EVID-12 / SLM-543).

External formats are projections only. Repository IDs, digests, and checker
capabilities remain authoritative — importing a PROV/SPDX/in-toto record never
upgrades semantic authority.
"""

from __future__ import annotations

from slm_training.evidence_interop.authority import (
    AuthorityRecord,
    AuthorityRoundtripError,
    ProjectionView,
    UnsupportedFieldError,
    authority_ids_equal,
    extract_authority,
)
from slm_training.evidence_interop.export_intoto import export_intoto
from slm_training.evidence_interop.export_oci import export_oci_referrer
from slm_training.evidence_interop.export_prov import export_prov_o
from slm_training.evidence_interop.export_rocrate import export_ro_crate
from slm_training.evidence_interop.export_spdx import export_spdx
from slm_training.evidence_interop.import_validate import (
    import_intoto,
    import_oci_referrer,
    import_projection,
    import_prov_o,
    import_ro_crate,
    import_spdx,
)
from slm_training.evidence_interop.profile import (
    AUTHORITY_FIELDS,
    ISSUE,
    KEY,
    PROFILE_ID,
    PROFILE_VERSION,
    SUPPORTED_FORMATS,
)
from slm_training.evidence_interop.roundtrip import roundtrip, roundtrip_all

__all__ = [
    "AUTHORITY_FIELDS",
    "ISSUE",
    "KEY",
    "PROFILE_ID",
    "PROFILE_VERSION",
    "SUPPORTED_FORMATS",
    "AuthorityRecord",
    "AuthorityRoundtripError",
    "ProjectionView",
    "UnsupportedFieldError",
    "authority_ids_equal",
    "extract_authority",
    "export_intoto",
    "export_oci_referrer",
    "export_prov_o",
    "export_ro_crate",
    "export_spdx",
    "import_intoto",
    "import_oci_referrer",
    "import_projection",
    "import_prov_o",
    "import_ro_crate",
    "import_spdx",
    "roundtrip",
    "roundtrip_all",
]
