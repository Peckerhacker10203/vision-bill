"""Lossless monetary storage and current review decisions."""

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

MONEY = {
    "receipts": ("subtotal", "discount_total", "tax_total", "tip", "total"),
    "line_items": ("unit_price", "total_price"),
    "taxes": ("amount",),
}


def upgrade() -> None:
    for table, columns in MONEY.items():
        for column in columns:
            op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE NUMERIC")
    op.execute("ALTER TABLE receipts ADD COLUMN overlap_decisions JSONB NOT NULL DEFAULT '[]'")
    op.execute("ALTER TABLE receipts ADD COLUMN acknowledgements JSONB NOT NULL DEFAULT '[]'")


def downgrade() -> None:
    # Refuse, never round or discard newer review decisions.
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM receipts WHERE
        overlap_decisions<>'[]'::jsonb OR acknowledgements<>'[]'::jsonb) THEN
        RAISE EXCEPTION 'review decision downgrade refused'; END IF; END $$""")
    for table, columns in MONEY.items():
        for column in columns:
            op.execute(f"""DO $$ BEGIN IF EXISTS (SELECT 1 FROM {table} WHERE
                {column}<>round({column},2) OR abs({column})>=1000000000000) THEN
                RAISE EXCEPTION 'precision losing monetary downgrade refused';
                END IF; END $$""")
    for table, columns in MONEY.items():
        for column in columns:
            op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE NUMERIC(14,2)")
    op.execute("ALTER TABLE receipts DROP COLUMN acknowledgements, DROP COLUMN overlap_decisions")
