"""Receipt CRUD + verify over live HTTP (uses the first registered fixture)."""

import pytest

from e2e.discovery import first_case
from e2e.helpers import (
    API,
    receipt_write,
    register_user,
    wait_until,
)

pytestmark = pytest.mark.e2e


def _make_receipt(client) -> str:
    case = first_case()
    assert case is not None, "no e2e fixtures registered"
    with case.image_path.open("rb") as fh:
        response = client.post(
            f"{API}/images", files={"receipt": (case.image_path.name, fh, "image/png")}
        )
    assert response.status_code == 202
    image_id = response.json()["id"]

    def check():
        row = client.get(f"{API}/images/{image_id}").json()
        return row["receipt_id"] if row["status"] == "analyzed" else None

    return wait_until(check, timeout_s=60.0)


def test_receipt_crud(http_factory) -> None:
    client, _ = register_user(http_factory, "crud")
    receipt_id = _make_receipt(client)

    # GET detail
    detail = client.get(f"{API}/receipts/{receipt_id}")
    assert detail.status_code == 200

    # PUT update (change merchant_name, keep required fields valid)
    current = detail.json()
    update = receipt_write(current, merchant_name="Renamed Merchant")
    put = client.put(f"{API}/receipts/{receipt_id}", json=update)
    assert put.status_code == 200
    assert put.json()["merchant_name"] == "Renamed Merchant"

    # DELETE
    delete = client.delete(f"{API}/receipts/{receipt_id}")
    assert delete.status_code == 200
    assert delete.json() == {"deleted": receipt_id}
    assert client.get(f"{API}/receipts/{receipt_id}").status_code == 404
    assert client.delete(f"{API}/receipts/{receipt_id}").status_code == 404
