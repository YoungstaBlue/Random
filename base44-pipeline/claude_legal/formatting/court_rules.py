"""Court Rules Compliance (Module 5).

Applies court-specific formatting templates: caption, headers/footers, signature
block, and certificate of service. Produces a :class:`FormattedDocument`.
"""
from __future__ import annotations

from datetime import date

from ..schemas import CourtFormatConfig, FormattedDocument


class CourtRulesFormatter:
    def format(self, cfg: CourtFormatConfig, body: str,
               served_on: str = "opposing counsel of record",
               service_date: date | None = None) -> FormattedDocument:
        service_date = service_date or date.today()

        caption = self._caption(cfg)
        signature = self._signature(cfg)
        cert = self._certificate_of_service(served_on, service_date)
        full = "\n\n".join([caption, cfg.document_title.upper(), body, signature, cert])
        return FormattedDocument(
            caption=caption,
            body=body,
            signature_block=signature,
            certificate_of_service=cert,
            full_text=full,
        )

    @staticmethod
    def _caption(cfg: CourtFormatConfig) -> str:
        district = f"\nFOR THE {cfg.district.upper()}" if cfg.district else ""
        parties = (
            f"{(cfg.plaintiff or '[PLAINTIFF]').upper()},\n"
            "                    Plaintiff,\n"
            "        v.\n"
            f"{(cfg.defendant or '[DEFENDANT]').upper()},\n"
            "                    Defendant."
        )
        case_no = f"Case No. {cfg.case_number}" if cfg.case_number else "Case No. __________"
        return (
            f"{cfg.court_name.upper()}{district}\n\n"
            f"{parties}\n\n"
            f"{'':>40}{case_no}\n"
            f"{'-' * 72}"
        )

    @staticmethod
    def _signature(cfg: CourtFormatConfig) -> str:
        name = cfg.attorney_name or "[ATTORNEY NAME]"
        bar = f"\nBar No. {cfg.attorney_bar}" if cfg.attorney_bar else "\nBar No. __________"
        return (
            "Respectfully submitted,\n\n"
            "_______________________________\n"
            f"{name}{bar}\n"
            "Attorney for the Filing Party"
        )

    @staticmethod
    def _certificate_of_service(served_on: str, service_date: date) -> str:
        return (
            "CERTIFICATE OF SERVICE\n\n"
            f"I hereby certify that on {service_date.isoformat()}, a true and correct "
            f"copy of the foregoing was served upon {served_on} via the Court's "
            "electronic filing system.\n\n"
            "_______________________________\n"
            "Signature"
        )
