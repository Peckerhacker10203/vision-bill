"""Explicit test-only dedication for existing owned handoff lifecycle fixtures."""
from vision_bill.model.handoff import HandoffBinding
from vision_bill.service.gpu_handoff import ModelHandoff


def install_owned_bindings(monkeypatch):
    initialize = ModelHandoff.__init__
    def initialize_owned(self, path, provider, ocr):
        initialize(self, path, provider, ocr)
        try:
            resources = self.resources
        except ValueError:
            return  # Identity validation scenarios intentionally have no valid binding.
        if not self.binding_path.exists():
            self.binding_path.parent.mkdir(parents=True, exist_ok=True)
            self.binding_path.write_text(HandoffBinding(
                coordination_path=str(self.path.resolve()), resources=resources,
                model_ids=["own", "formatter", "fixture"],
                dedicated_coordinated_requests=True, external_eager_loading_disabled=True,
            ).model_dump_json())
    monkeypatch.setattr(ModelHandoff, "__init__", initialize_owned)
