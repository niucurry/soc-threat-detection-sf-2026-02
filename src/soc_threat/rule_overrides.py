from __future__ import annotations


def suspicious_rule_sql(alias: str = "i") -> str:
    return f"""
        {alias}.product_name = 'Symantec Data Loss Prevention'
        OR (
            {alias}.product_name = 'Duo'
            AND CONTAINS(LOWER({alias}.message_sanitized), 'invalid_passcode')
            AND CONTAINS(LOWER({alias}.message_sanitized), 'auth_failure')
        )
    """


def suspicious_rule_name_sql(alias: str = "i") -> str:
    condition = suspicious_rule_sql(alias)
    return f"""
        CASE
            WHEN {alias}.product_name = 'Symantec Data Loss Prevention'
                THEN 'symantec_dlp'
            WHEN {condition} THEN 'duo_invalid_passcode'
        END
    """
