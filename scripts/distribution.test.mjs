import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  readdirSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { spawnSync } from "node:child_process";
import {
  validateArtifacts,
  renderBundle,
  writeBundle,
} from "./distribution.mjs";

const hash = (value) => createHash("sha256").update(value).digest("hex");
const digest = (value) => `sha256:${hash(value)}`;
const image = (name) => ({
  reference: `ghcr.io/maintainer/vision-bill-${name}@${digest(name)}`,
  config_digest: digest(`${name}-config`),
});
const license = {
  spdx: "Apache-2.0",
  url: "https://www.apache.org/licenses/LICENSE-2.0",
  file: "llm/LICENSE",
  redistribution_allowed: true,
};
function fixture() {
  const assets = {};
  const bytes = {};
  function asset(path, value) {
    const content = typeof value === "string" ? value : JSON.stringify(value);
    bytes[path] = content;
    assets[path] = {
      sha256: hash(content),
      size_bytes: Buffer.byteLength(content),
    };
    return path;
  }
  const manifest = {
    schema_version: 1,
    version: "1.2.3",
    platform: "linux/amd64",
    source: {
      repository: "maintainer/vision-bill",
      commit: hash("source").slice(0, 40),
    },
    images: {
      app: image("app"),
      postgres: image("postgres"),
      "ollama-cuda": image("ollama-cuda"),
      "ollama-rocm": image("ollama-rocm"),
      "ocr-cuda": image("ocr-cuda"),
      "ocr-rocm": image("ocr-rocm"),
    },
    llm: {
      model: "llama3.2-vision:11b",
      digest: digest("actual-model"),
      license,
      provenance: {
        url: "https://ollama.com/library/llama3.2-vision",
        revision: hash("model-revision"),
      },
    },
    verification: { app_checks: "checks.json" },
    native: {},
    ollama: {},
    assets,
  };
  asset("checks.json", {
    passed: true,
    source_commit: manifest.source.commit,
    checks: ["backend", "frontend", "e2e-backend", "e2e-frontend"],
  });
  asset("llm/LICENSE", "synthetic test-only license text");
  for (const accelerator of ["cuda", "rocm"]) {
    const selected = manifest.images[`ollama-${accelerator}`],
      base = image(`ollama-base-${accelerator}`).reference;
    manifest.ollama[accelerator] = {
      base,
      capability: asset(`llm/${accelerator}-backend.json`, {
        scope: "backend-package-only",
        accelerator,
        base_reference: base,
        image_identity: selected.reference,
        config_digest: selected.config_digest,
        backend_libraries: {
          [`/usr/lib/ollama/${accelerator}/libggml-${accelerator === "cuda" ? "cuda" : "hip"}.so`]:
            hash(`${accelerator}-elf`),
        },
        dependencies_resolved: true,
        target_markers: accelerator === "rocm" ? ["gfx1030"] : [],
        target_files:
          accelerator === "rocm"
            ? { "/usr/lib/ollama/rocm/libggml-hip.so": hash("rocm-elf") }
            : {},
      }),
    };
    const config = JSON.stringify({
      architecture: "amd64",
      os: "linux",
      config: { Cmd: ["python", "-m", "ocr_service.entrypoint"] },
    });
    manifest.images[`ocr-${accelerator}`].config_digest = digest(config);
    const nativeLicense = {
      spdx: "Apache-2.0",
      text_file: "LICENSE",
      redistribution_allowed: true,
    };
    const provenance = {
      url: "https://huggingface.co/PaddlePaddle",
      revision: hash("revision"),
    };
    const archive = (name) => {
      const files = {
        LICENSE: { sha256: hash("license-text"), size_bytes: 12 },
        [name]: { sha256: hash(name), size_bytes: 15 },
      };
      return {
        source: `https://models.vision-bill.org/${name}.tar`,
        sha256: hash(`${name}-tar`),
        size_bytes: 128,
        unpacked_bytes: 27,
        files,
        tree_sha256: hash(
          JSON.stringify(
            Object.fromEntries(
              Object.entries(files)
                .sort()
                .map(([path, record]) => [path, record.sha256]),
            ),
          ),
        ),
        license: nativeLicense,
        provenance,
      };
    };
    const models = Object.fromEntries(
      [
        "PP-DocLayoutV3",
        "PP-LCNet_x1_0_doc_ori",
        "UVDoc",
        "PaddleOCR-VL-1.6-0.9B",
      ].map((name) => [name, archive(name)]),
    );
    const fixtures = archive("upright");
    fixtures.cases = ["upright", "rotated", "warped", "table", "formula"].map(
      (kind) => ({
        kind,
        path: kind,
        sha256: hash(kind),
        required_labels: ["table", "formula"].includes(kind) ? [kind] : [],
        license: nativeLicense,
        provenance,
      }),
    );
    for (const kind of ["rotated", "warped", "table", "formula"])
      fixtures.files[kind] = { sha256: hash(kind), size_bytes: 15 };
    fixtures.unpacked_bytes = 87;
    fixtures.tree_sha256 = hash(
      JSON.stringify(
        Object.fromEntries(
          Object.entries(fixtures.files)
            .sort()
            .map(([path, record]) => [path, record.sha256]),
        ),
      ),
    );
    const bundle = {
      schema_version: 1,
      accelerator,
      gpu_index: 0,
      image_identity: manifest.images[`ocr-${accelerator}`].reference,
      image_config_sha256: hash(config),
      models,
      fixtures,
      profile_sha256: hash("profile"),
      source_files: {
        "profile.json": hash("profile"),
        "runtime-lock.json": hash("source-lock"),
      },
      expected_paddle_commit: hash("paddle-source").slice(0, 40),
      expected_packages: {
        paddleocr: "3.7.0",
        paddlex: "3.7.2",
        "paddlepaddle-gpu": "3.3.1",
      },
    };
    const seal = {
      image_identity: bundle.image_identity,
      paddle_identity: `paddle:${bundle.expected_paddle_commit}`,
      model_content: Object.fromEntries(
        Object.entries(models).map(([name, row]) => [name, row.tree_sha256]),
      ),
      qualified_target:
        accelerator === "rocm"
          ? { gfx_target: "gfx1030", name: "Radeon RX 6800" }
          : { name: "NVIDIA RTX 3090" },
      smoke_sha256: fixtures.cases.find((row) => row.kind === "upright").sha256,
    };
    const report = {
      qualified: true,
      stage: "complete",
      accelerator,
      image_identity: bundle.image_identity,
      profile_id: hash("profile-id"),
      runtime_fingerprint: hash("runtime-id"),
      model_content: structuredClone(seal.model_content),
      manifest_sha256: hash(JSON.stringify(seal)),
      device:
        accelerator === "rocm"
          ? { gfx_target: "gfx1030", name: "Radeon RX 6800" }
          : { name: "NVIDIA RTX 3090" },
      qualified_targets: [
        accelerator === "rocm" ? "gfx1030" : "NVIDIA RTX 3090",
      ],
      cases: ["upright", "rotated", "warped", "table", "formula"].flatMap(
        (kind) =>
          [0, 1].map((fresh_worker) => ({
            kind,
            fresh_worker,
            passed: true,
            source_sha256: fixtures.cases.find((row) => row.kind === kind)
              .sha256,
          })),
      ),
    };
    manifest.native[accelerator] = {
      bundle: asset(`${accelerator}/bundle.json`, bundle),
      image_config: asset(`${accelerator}/image-config.json`, config),
      qualification: asset(`${accelerator}/qualification.json`, report),
      seal: asset(`${accelerator}/qualification-manifest.json`, seal),
      ...(accelerator === "cuda"
        ? {
            cuda_lock: asset("cuda/lock-attestation.json", {
              index:
                "https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/",
              index_sha256: hash("index"),
              wheel_url:
                "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl",
              wheel_sha256: hash("wheel"),
              metadata_sha256: hash("metadata"),
              metadata_name: "paddlepaddle-gpu",
              metadata_version: "3.3.1",
              requires_dist: ["numpy"],
              lock_sha256: hash("resolved-lock"),
              platform: "linux-x86_64-cp312",
            }),
          }
        : {}),
    };
    if (accelerator === "cuda")
      asset("cuda/requirements.cuda.lock", "resolved-lock");
  }
  return { manifest, bytes };
}
const variants = [
  "app",
  "ocr-cuda",
  "ocr-rocm",
  "combined-cuda",
  "combined-rocm",
];
function temp(t) {
  const path = mkdtempSync(join(tmpdir(), "vb-distribution-"));
  t.after(() => rmSync(path, { recursive: true, force: true }));
  return path;
}
function seed(path, bytes) {
  for (const [name, value] of Object.entries(bytes)) {
    mkdirSync(resolve(path, name, ".."), { recursive: true });
    writeFileSync(join(path, name), value);
  }
}

// A wrong digest or omitted release evidence must make production validation fail.
test("artifact validation rejects absent, mutable, placeholder and contradictory identities", () => {
  const { manifest } = fixture();
  assert.equal(validateArtifacts(manifest), manifest);
  for (const change of [
    (m) => delete m.source.commit,
    (m) => (m.source.commit = "0".repeat(40)),
    (m) => (m.images.app.reference = "ghcr.io/maintainer/app:latest"),
    (m) => (m.images.app.config_digest = "sha256:" + "0".repeat(64)),
    (m) => (m.images.app.reference = `ghcr.io/example/app@${digest("a")}`),
    (m) =>
      (m.images.app.reference =
        "ghcr.io/maintainer/app@sha256:" + "0".repeat(64)),
    (m) => delete m.native.cuda.qualification,
    (m) => delete m.native.cuda.cuda_lock,
    (m) => (m.llm.digest = "sha256:REPLACE_ME"),
    (m) => (m.llm.license.spdx = "TBD"),
    (m) => (m.assets["../outside"] = { sha256: hash("x"), size_bytes: 1 }),
    (m) => (m.images.app.password = "secret"),
    (m) => (m.unexpected = true),
  ]) {
    const bad = structuredClone(manifest);
    change(bad);
    assert.throws(() => validateArtifacts(bad));
  }
});

test("bundles are deterministic, isolated, immutable and contain only prebuilt images", () => {
  const { manifest } = fixture();
  for (const variant of variants) {
    const rendered = renderBundle(manifest, variant);
    assert.deepEqual(
      rendered,
      renderBundle(structuredClone(manifest), variant),
    );
    const reorder = (value) =>
      Array.isArray(value)
        ? value.map(reorder)
        : value && typeof value === "object"
          ? Object.fromEntries(
              Object.entries(value)
                .reverse()
                .map(([key, child]) => [key, reorder(child)]),
            )
          : value;
    assert.deepEqual(rendered, renderBundle(reorder(manifest), variant));
    const compose = JSON.parse(rendered.files["compose.yaml"]);
    const content = rendered.files["compose.yaml"];
    assert.doesNotMatch(
      content,
      /"build"|container_name|"external"|\/workspace|docker.sock|POSTGRES_PASSWORD"|AUTH__SECRET_KEY/,
    );
    for (const service of Object.values(compose.services))
      assert.match(service.image, /@sha256:[a-f0-9]{64}$/);
    for (const service of Object.values(compose.services))
      for (const mount of service.volumes ?? [])
        if (mount.type === "bind") assert.match(mount.source, /^\.\/release\//);
    if (variant === "app") {
      assert.deepEqual(Object.keys(compose.services).sort(), [
        "app",
        "config-init",
        "db",
      ]);
      assert.match(content, /LLM_HOST:\?/);
      assert.match(content, /LLM_MODEL:\?/);
      assert.equal(
        compose.services.app.environment.OCR__ENDPOINT,
        "${OCR_ENDPOINT:-}",
      );
      assert.doesNotMatch(content, /ollama:11434|installation.lock/);
    } else {
      const combined = variant.startsWith("combined");
      assert.equal(
        compose.services.ocr.environment.OCR_SERVICE_MODE,
        combined ? "managed" : "${OCR_SERVICE_MODE:-standalone}",
      );
      assert.equal(!!compose.services.ocr.ports, !combined);
      assert.equal(!!compose.services.app, combined);
      assert.equal(!!compose.services.ollama, combined);
      assert.equal(compose.services["ocr-init"].restart, "no");
      assert.equal(compose.services["ocr-init"].init, false);
      assert.equal(compose.services.ocr.init, false);
      if (variant.endsWith("rocm"))
        assert.equal(compose.services.ocr.group_add, undefined);
      assert.deepEqual(compose.services.ocr.depends_on["ocr-init"], {
        condition: "service_completed_successfully",
      });
      assert.match(
        compose.services["ocr-init"].command.join(" "),
        /--image-id sha256:[a-f0-9]{64}/,
      );
      assert.equal(
        compose.services["ocr-init"].environment.OCR_INSTALLATION_LOCK_PATH,
        "/coord/installation.lock",
      );
      assert.equal(
        compose.services.ocr.environment.OCR_INSTALLATION_LOCK_PATH,
        "/coord/installation.lock",
      );
      if (combined) {
        assert.equal(
          compose.services.ollama.image,
          manifest.images[`ollama-${variant.split("-").at(-1)}`].reference,
        );
        assert.equal(compose.services.ollama.group_add, undefined);
        assert.equal(compose.services.ollama.init, false);
        assert.deepEqual(compose.services.ollama.depends_on["ocr-init"], {
          condition: "service_completed_successfully",
        });
        assert.equal(
          compose.services.ollama.environment.OCR_INSTALLATION_LOCK_PATH,
          "/coord/installation.lock",
        );
        assert.equal(compose.services.ollama.ports, undefined);
        assert.equal(compose.services.app.depends_on.ollama, undefined);
        assert.deepEqual(compose.services.app.depends_on["model-init"], {
          condition: "service_completed_successfully",
        });
        assert.match(
          compose.services["model-init"].command.join(" "),
          new RegExp(`--expected-digest ${manifest.llm.digest}`),
        );
        assert.match(
          compose.services["model-init"].command.join(" "),
          new RegExp(`--bundled-model ${manifest.llm.model}`),
        );
        const modelArgs = compose.services["model-init"].command;
        assert.equal(
          modelArgs[modelArgs.indexOf("--bundled-model") + 1],
          manifest.llm.model,
        );
        assert.ok(
          !compose.services["config-init"].command.includes("--bundled-model"),
        );
        assert.equal(
          compose.services.app.environment.OCR__ENDPOINT,
          "http://ocr:8082",
        );
      }
      assert.ok(
        rendered.assets.every(
          (name) =>
            !name.startsWith(variant.endsWith("cuda") ? "rocm/" : "cuda/"),
        ),
      );
    }
  }
  assert.throws(() => renderBundle(manifest, "latest"));
});

test("materialization verifies raw OCI config and qualification before atomic complete bundle publication", (t) => {
  const directory = temp(t);
  const { manifest, bytes } = fixture();
  seed(directory, bytes);
  const output = join(directory, "out");
  writeBundle(manifest, "combined-rocm", output, directory);
  const checksums = readFileSync(join(output, "SHA256SUMS"), "utf8");
  for (const line of checksums.trim().split("\n")) {
    const [expected, path] = line.split("  ");
    assert.equal(hash(readFileSync(join(output, path))), expected);
  }
  const metadata = JSON.parse(readFileSync(join(output, "release.json")));
  assert.equal(metadata.source.commit, manifest.source.commit);
  assert.equal(metadata.variant, "combined-rocm");
  assert.throws(() =>
    writeBundle(manifest, "combined-rocm", output, directory),
  );
  writeFileSync(join(directory, "rocm/image-config.json"), "{}");
  const failed = join(directory, "failed");
  assert.throws(
    () => writeBundle(manifest, "ocr-rocm", failed, directory),
    /identity|hash|digest/i,
  );
  assert.ok(!readdirSync(directory).includes("failed"));
});

test("same-hash evidence with wrong config or incomplete hardware qualification fails closed", (t) => {
  for (const mutate of [
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.manifest_sha256 = hash("wrong-seal");
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
    (m, b) => {
      m.images["ocr-rocm"].config_digest = digest("different-config");
    },
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.qualified = false;
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.device.gfx_target = "gfx1100";
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.cases.pop();
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
  ]) {
    const path = temp(t);
    const { manifest, bytes } = fixture();
    mutate(manifest, bytes);
    seed(path, bytes);
    assert.throws(() =>
      writeBundle(manifest, "ocr-rocm", join(path, "out"), path),
    );
  }
});

test("external native bundle requires complete model, fixture and licensed archive inventories", (t) => {
  for (const mutate of [
    (b) => delete b.models.UVDoc,
    (b) => delete b.models.UVDoc.license,
    (b) => (b.models.UVDoc.tree_sha256 = hash("wrong-tree")),
    (b) =>
      (b.models.UVDoc.files["../unsafe"] = {
        sha256: hash("unsafe"),
        size_bytes: 1,
      }),
    (b) => b.fixtures.cases.pop(),
    (b) => (b.fixtures.cases[4].required_labels = []),
    (b) => (b.expected_packages.paddleocr = "3.6.0"),
  ]) {
    const path = temp(t);
    const { manifest, bytes } = fixture();
    const name = manifest.native.rocm.bundle;
    const bundle = JSON.parse(bytes[name]);
    mutate(bundle);
    bytes[name] = JSON.stringify(bundle);
    manifest.assets[name] = {
      sha256: hash(bytes[name]),
      size_bytes: bytes[name].length,
    };
    seed(path, bytes);
    assert.throws(() =>
      writeBundle(manifest, "ocr-rocm", join(path, "out"), path),
    );
  }
});

test("CLI generates every variant and Docker Compose validates effective config without services", (t) => {
  const directory = temp(t);
  const { manifest, bytes } = fixture();
  seed(directory, bytes);
  writeFileSync(join(directory, "artifacts.json"), JSON.stringify(manifest));
  const env = {
    ...process.env,
    LLM_PROVIDER: "ollama",
    LLM_HOST: "http://operator-llm:11434",
    LLM_MODEL: "operator-model",
  };
  delete env.OCR_RENDER_GID;
  delete env.OCR_VIDEO_GID;
  delete env.OCR_SERVICE_MODE;
  for (const variant of variants) {
    const output = join(directory, variant);
    const generated = spawnSync(
      process.execPath,
      [
        "scripts/distribution.mjs",
        "--manifest",
        join(directory, "artifacts.json"),
        "--variant",
        variant,
        "--output",
        output,
      ],
      { encoding: "utf8", env },
    );
    assert.equal(generated.status, 0, generated.stderr);
    const composed = spawnSync(
      "docker",
      [
        "compose",
        "-f",
        join(output, "compose.yaml"),
        "config",
        "--format",
        "json",
      ],
      { encoding: "utf8", env },
    );
    assert.equal(composed.status, 0, composed.stderr);
    const effective = JSON.parse(composed.stdout);
    if (effective.services.ocr)
      assert.equal(
        effective.services.ocr.environment.OCR_SERVICE_MODE,
        variant.startsWith("combined") ? "managed" : "standalone",
      );
    if (variant.startsWith("ocr-")) {
      const managed = spawnSync(
        "docker",
        [
          "compose",
          "-f",
          join(output, "compose.yaml"),
          "config",
          "--format",
          "json",
        ],
        { encoding: "utf8", env: { ...env, OCR_SERVICE_MODE: "managed" } },
      );
      assert.equal(managed.status, 0, managed.stderr);
      assert.equal(
        JSON.parse(managed.stdout).services.ocr.environment.OCR_SERVICE_MODE,
        "managed",
      );
    }
    for (const name of ["ocr-init", "ocr", "ollama"])
      if (effective.services[name])
        assert.equal(effective.services[name].init, false);
    assert.ok(
      !Object.values(effective.services).some((service) => service.build),
    );
  }
});

test("derived Ollama PID1 supervisor blocks live and crash-marked initialization until confirmed cleanup", async (t) => {
  const dockerfile = readFileSync("distribution/ollama/Dockerfile", "utf8");
  assert.match(dockerfile, /python3/);
  assert.match(dockerfile, /ocr_service\/supervision.py/);
  assert.match(dockerfile, /apt-get install[^\n]*util-linux/);
  assert.match(dockerfile, /command -v flock/);
  const directory = mkdtempSync(join(tmpdir(), "vb-distribution-lock-"));
  const containers = new Set();
  const coord = join(directory, "coord"),
    runtime = join(directory, "runtime");
  mkdirSync(coord);
  mkdirSync(runtime);
  t.after(() => {
    for (const id of containers) spawnSync("docker", ["rm", "-f", id]);
    rmSync(directory, { recursive: true, force: true });
  });
  writeFileSync(
    join(runtime, "entrypoint.sh"),
    readFileSync("distribution/ollama/entrypoint.sh"),
  );
  writeFileSync(
    join(runtime, "supervision.py"),
    readFileSync("paddleocr-service/ocr_service/supervision.py"),
  );
  writeFileSync(join(runtime, "ollama"), "#!/bin/sh\nexec sleep 300\n", {
    mode: 0o755,
  });
  const docker = (args) => {
    const result = spawnSync("docker", args, {
      encoding: "utf8",
      timeout: 60000,
    });
    assert.equal(result.error, undefined, result.error?.message);
    return result;
  };
  const success = (args) => {
    const result = docker(args);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    return result.stdout.trim();
  };
  const image = "python:3.12-slim";
  if (docker(["image", "inspect", image]).status !== 0)
    success(["pull", image]);
  const mounts = [
    "--init=false",
    "--mount",
    `type=bind,src=${coord},dst=/coord`,
    "--mount",
    `type=bind,src=${runtime},dst=/runtime,readonly`,
    "--mount",
    `type=bind,src=${runtime}/ollama,dst=/bin/ollama,readonly`,
  ];
  const marker = join(coord, "installation.active-ollama.json");
  const waitFor = async (condition) => {
    const until = Date.now() + 5000;
    while (!condition()) {
      assert.ok(Date.now() < until, "supervisor state timed out");
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
  };
  const start = async () => {
    const id = success([
      "create",
      ...mounts,
      image,
      "sh",
      "/runtime/entrypoint.sh",
      "serve",
    ]);
    assert.match(id, /^[a-f0-9]{64}$/);
    containers.add(id);
    success(["start", id]);
    await waitFor(
      () =>
        spawnSync("docker", [
          "exec",
          id,
          "test",
          "-f",
          "/coord/installation.active-ollama.json",
        ]).status === 0,
    );
    return id;
  };
  const runOwned = (command) => {
    const id = success(["create", ...mounts, image, ...command]);
    containers.add(id);
    try {
      return docker(["start", "--attach", id]);
    } finally {
      success(["rm", "-f", id]);
      containers.delete(id);
    }
  };
  const attemptInit = () =>
    runOwned([
      "python3",
      "/runtime/supervision.py",
      "--role",
      "initializer",
      "--lock",
      "/coord/installation.lock",
      "--",
      "sh",
      "-c",
      "touch /coord/should-not-run",
    ]);
  const { existsSync } = await import("node:fs");
  let id = await start();
  assert.notEqual(attemptInit().status, 0);
  assert.ok(!existsSync(join(coord, "should-not-run")));
  assert.notEqual(
    docker([
      "run",
      "--rm",
      ...mounts,
      image,
      "flock",
      "--exclusive",
      "--nonblock",
      "/coord/installation.lock",
      "true",
    ]).status,
    0,
  );
  success(["stop", "--time", "10", id]);
  assert.ok(
    !existsSync(marker),
    "ordinary termination must reap before clearing marker",
  );
  success(["rm", id]);
  containers.delete(id);
  id = await start();
  success(["kill", "--signal", "KILL", id]);
  success(["wait", id]);
  assert.ok(existsSync(marker), "hard death must leave durable marker");
  assert.notEqual(attemptInit().status, 0);
  assert.notEqual(
    runOwned(["sh", "/runtime/entrypoint.sh", "serve"]).status,
    0,
  );
  assert.ok(!existsSync(join(coord, "should-not-run")));
  success(["rm", id]);
  containers.delete(id);
  assert.notEqual(
    docker(["inspect", id]).status,
    0,
    "positive external removal evidence precedes test-only recovery",
  );
  // Only this isolated test's known container existed. Production never clears
  // markers automatically; operator recovery requires all owned peers removed.
  success([
    "run",
    "--rm",
    ...mounts,
    image,
    "rm",
    "/coord/installation.active-ollama.json",
  ]);
  id = await start();
  success(["stop", "--time", "10", id]);
  assert.ok(!existsSync(marker));
  success(["rm", id]);
  containers.delete(id);
  const peer = success([
    "create",
    ...mounts,
    image,
    "python3",
    "/runtime/supervision.py",
    "--role",
    "ocr",
    "--lock",
    "/coord/installation.lock",
    "--",
    "sleep",
    "300",
  ]);
  containers.add(peer);
  success(["start", peer]);
  await waitFor(() => existsSync(join(coord, "installation.active-ocr.json")));
  id = await start(); // A genuinely live OCR peer can share serving ownership.
  success(["stop", "--time", "10", id]);
  success(["rm", id]);
  containers.delete(id);
  success(["kill", "--signal", "KILL", peer]);
  success(["wait", peer]);
  assert.notEqual(
    runOwned(["sh", "/runtime/entrypoint.sh", "serve"]).status,
    0,
    "stale peer marker must block Ollama",
  );
  assert.notEqual(attemptInit().status, 0);
  success(["rm", peer]);
  containers.delete(peer);
  assert.notEqual(docker(["inspect", peer]).status, 0);
  success([
    "run",
    "--rm",
    ...mounts,
    image,
    "rm",
    "/coord/installation.active-ocr.json",
  ]);
});

test("derived Docker context includes exact canonical supervisor and entrypoint without a native build", (t) => {
  const output = join(temp(t), "context");
  const probe = spawnSync(
    "docker",
    [
      "buildx",
      "build",
      "--file",
      "-",
      "--output",
      `type=local,dest=${output}`,
      ".",
    ],
    {
      encoding: "utf8",
      input:
        "FROM scratch\nCOPY distribution/ollama/entrypoint.sh /entrypoint.sh\nCOPY paddleocr-service/ocr_service/supervision.py /supervision.py\nCOPY distribution/ollama/probe.py /probe.py\n",
      timeout: 60000,
    },
  );
  assert.equal(probe.status, 0, probe.stderr);
  assert.equal(
    hash(readFileSync(join(output, "supervision.py"))),
    hash(readFileSync("paddleocr-service/ocr_service/supervision.py")),
  );
  assert.equal(
    hash(readFileSync(join(output, "entrypoint.sh"))),
    hash(readFileSync("distribution/ollama/entrypoint.sh")),
  );
  assert.equal(
    hash(readFileSync(join(output, "probe.py"))),
    hash(readFileSync("distribution/ollama/probe.py")),
  );
});

test("manual distribution workflow gates fork/exact-SHA, native qualification, publication and assets in order", () => {
  const workflow = readFileSync(
    ".github/workflows/distribution-release.yml",
    "utf8",
  );
  assert.match(workflow, /workflow_dispatch:/);
  assert.doesNotMatch(workflow, /^  (push|pull_request|schedule):/m);
  assert.match(workflow, /DISTRIBUTION_FORK_REPOSITORY/);
  assert.match(workflow, /github\.sha/);
  assert.match(workflow, /needs: \[verify, prepare\]/);
  assert.ok(
    workflow.indexOf("Qualify exact native image") <
      workflow.indexOf("Publish verified images"),
  );
  assert.ok(
    workflow.indexOf("Verify registry pullability") <
      workflow.indexOf("Publish release assets last"),
  );
  assert.match(
    readFileSync(".github/workflows/ci.yml", "utf8"),
    /make distribution-test/,
  );
});

test("maintainer request rejects foreign forks, events, mutable source and missing real inputs", async () => {
  const release = await import("./distribution-release.mjs");
  const env = {
    GITHUB_EVENT_NAME: "workflow_dispatch",
    GITHUB_REPOSITORY: "maintainer/vision-bill",
    DISTRIBUTION_FORK_REPOSITORY: "maintainer/vision-bill",
    GITHUB_SHA: hash("source").slice(0, 40),
    DISTRIBUTION_SOURCE_SHA: hash("source").slice(0, 40),
    DISTRIBUTION_VERSION: "1.2.3",
    DISTRIBUTION_VARIANT: "combined-rocm",
    GITHUB_RUN_ID: "123",
  };
  const request = release.releaseRequest(env);
  assert.equal(request.variant, "combined-rocm");
  for (const change of [
    { GITHUB_EVENT_NAME: "push" },
    { DISTRIBUTION_FORK_REPOSITORY: "" },
    { GITHUB_REPOSITORY: "upstream/vision-bill" },
    { DISTRIBUTION_SOURCE_SHA: hash("other").slice(0, 40) },
    { DISTRIBUTION_VERSION: "latest" },
    { DISTRIBUTION_VARIANT: "all" },
  ])
    assert.throws(() => release.releaseRequest({ ...env, ...change }));
  const plan = {
    schema_version: 1,
    source: { repository: request.repository, commit: request.sha },
    version: request.version,
    platform: "linux/amd64",
    postgres: image("postgres").reference,
    ollama_base: {
      accelerator: "rocm",
      reference: image("ollama-base-rocm").reference,
    },
    llm: fixture().manifest.llm,
    rocm: {
      targets: ["gfx1030"],
      render_node: "/dev/dri/renderD128",
      render_gid: 109,
      video_gid: 44,
    },
  };
  assert.equal(release.validateReleaseInputs(plan, request), plan);
  for (const mutate of [
    (p) => (p.ollama_base = "ollama/ollama:latest"),
    (p) => (p.ollama_base.accelerator = "cuda"),
    (p) => delete p.llm.digest,
    (p) => (p.rocm.targets = ["any"]),
    (p) => (p.rocm.render_node = "/dev"),
    (p) => (p.source.commit = hash("old").slice(0, 40)),
  ]) {
    const bad = structuredClone(plan);
    mutate(bad);
    assert.throws(() => release.validateReleaseInputs(bad, request));
  }
  const rootPlan = structuredClone(plan);
  delete rootPlan.rocm.render_gid;
  delete rootPlan.rocm.video_gid;
  assert.equal(release.validateReleaseInputs(rootPlan, request), rootPlan);
});

test("combined bundles reject cross-accelerator or unsupported-target Ollama evidence and non-root native config", (t) => {
  for (const mutate of [
    (p) => (p.accelerator = "cuda"),
    (p) => (p.base_reference = image("foreign-base").reference),
    (p) => (p.dependencies_resolved = false),
    (p) => (p.target_markers = []),
    (p) =>
      (p.target_files = {
        "/usr/lib/ollama/cuda/libggml-cuda.so": hash("foreign"),
      }),
    (p) =>
      (p.backend_libraries = {
        "/usr/lib/ollama/cuda/libggml-cuda.so": hash("foreign"),
      }),
  ]) {
    const directory = temp(t),
      { manifest, bytes } = fixture(),
      name = manifest.ollama.rocm.capability;
    const proof = JSON.parse(bytes[name]);
    mutate(proof);
    bytes[name] = JSON.stringify(proof);
    manifest.assets[name] = {
      sha256: hash(bytes[name]),
      size_bytes: bytes[name].length,
    };
    seed(directory, bytes);
    assert.throws(() =>
      writeBundle(manifest, "combined-rocm", join(directory, "out"), directory),
    );
  }
  const directory = temp(t),
    { manifest, bytes } = fixture(),
    native = manifest.native.rocm;
  const config = JSON.parse(bytes[native.image_config]);
  config.config.User = "1000:1000";
  bytes[native.image_config] = JSON.stringify(config);
  manifest.images["ocr-rocm"].config_digest = digest(
    bytes[native.image_config],
  );
  const bundle = JSON.parse(bytes[native.bundle]);
  bundle.image_config_sha256 = hash(bytes[native.image_config]);
  bytes[native.bundle] = JSON.stringify(bundle);
  for (const name of [native.bundle, native.image_config])
    manifest.assets[name] = {
      sha256: hash(bytes[name]),
      size_bytes: bytes[name].length,
    };
  seed(directory, bytes);
  assert.throws(
    () => writeBundle(manifest, "ocr-rocm", join(directory, "out"), directory),
    /root/,
  );
});

test("publication cannot create assets after qualification, upload or pullability failure", async () => {
  const { publishVerified } = await import("./distribution-release.mjs");
  const { manifest } = fixture();
  for (const failure of ["verify", "image", "pull", null]) {
    const calls = [];
    const op = (method) => async () => {
      calls.push(method);
      if (method === failure) throw Error("gate failed");
    };
    const promise = publishVerified(manifest, "combined-cuda", {
      verify: op("verify"),
      image: op("image"),
      pull: op("pull"),
      assets: op("assets"),
    });
    if (failure) {
      await assert.rejects(promise, /gate failed/);
      assert.ok(!calls.includes("assets"));
    } else {
      await promise;
      assert.equal(calls.at(-1), "assets");
      assert.ok(calls.indexOf("image") > calls.indexOf("verify"));
      assert.ok(calls.indexOf("pull") > calls.lastIndexOf("image"));
    }
  }
});

test("failed qualification container creation never removes a preexisting named container", async () => {
  const { withOwnedContainer } = await import("./distribution-release.mjs");
  const removed = [],
    id = hash("owned-container");
  assert.throws(
    () =>
      withOwnedContainer(
        {
          create() {
            throw Error("name collision");
          },
          remove(value) {
            removed.push(value);
          },
        },
        () => {},
      ),
    /name collision/,
  );
  assert.deepEqual(removed, []);
  assert.throws(
    () =>
      withOwnedContainer(
        {
          create: () => id,
          remove(value) {
            removed.push(value);
          },
        },
        () => {
          throw Error("gate failed");
        },
      ),
    /gate failed/,
  );
  assert.deepEqual(removed, [id]);
});

test("partial publication can reuse only the exact already-qualified image digest", async () => {
  const { publishImageIfNeeded } = await import("./distribution-release.mjs");
  const expected = image("app");
  const copied = [];
  await publishImageIfNeeded(expected, {
    lookup: async () => null,
    copy: async () => copied.push("copy"),
  });
  await publishImageIfNeeded(expected, {
    lookup: async () => expected.reference.split("@")[1],
    copy: async () => copied.push("wrong"),
  });
  assert.deepEqual(copied, ["copy"]);
  await assert.rejects(
    publishImageIfNeeded(expected, {
      lookup: async () => digest("foreign"),
      copy: async () => copied.push("wrong"),
    }),
  );
  await assert.rejects(
    publishImageIfNeeded(expected, {
      lookup: async () => {
        throw Error("denied");
      },
      copy: async () => copied.push("wrong"),
    }),
    /denied/,
  );
  assert.deepEqual(copied, ["copy"]);
});

test("Ollama package probe verifies ELF backend and refuses wrong accelerator or absent AMD target without inference", () => {
  const probe = resolve("distribution/ollama/probe.py");
  const command =
    'mkdir -p /usr/lib/ollama/cuda /usr/lib/ollama/rocm; cp /lib/x86_64-linux-gnu/libc.so.6 /usr/lib/ollama/cuda/libggml-cuda.so; cp /lib/x86_64-linux-gnu/libc.so.6 /usr/lib/ollama/rocm/libggml-hip.so; python3 /probe.py "$@"';
  const execute = (args, source = command) =>
    spawnSync(
      "docker",
      [
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        `type=bind,src=${probe},dst=/probe.py,readonly`,
        "python:3.12-slim",
        "sh",
        "-c",
        source,
        "probe",
        ...args,
      ],
      { encoding: "utf8", timeout: 30000 },
    );
  const inspected = execute(["--accelerator", "cuda"]);
  assert.equal(inspected.status, 0, inspected.stderr);
  const report = JSON.parse(inspected.stdout);
  assert.equal(report.accelerator, "cuda");
  assert.equal(report.scope, "backend-package-only");
  assert.equal(report.dependencies_resolved, true);
  assert.ok(Object.keys(report.backend_libraries).length > 0);
  const missingTarget = execute([
    "--accelerator",
    "rocm",
    "--target",
    "gfx1030",
  ]);
  assert.equal(missingTarget.status, 1);
  const missingBackend = execute(
    ["--accelerator", "rocm"],
    command.replace(
      "cp /lib/x86_64-linux-gnu/libc.so.6 /usr/lib/ollama/rocm/libggml-hip.so;",
      "",
    ),
  );
  assert.equal(missingBackend.status, 1);
  const foreignTarget = execute(
    ["--accelerator", "rocm", "--target", "gfx1030"],
    command.replace(
      "python3 /probe.py",
      "printf gfx1030 >> /usr/lib/ollama/cuda/libggml-cuda.so; python3 /probe.py",
    ),
  );
  assert.equal(
    foreignTarget.status,
    1,
    "an unrelated CUDA library cannot attest the HIP target",
  );
  const hipTarget = execute(
    ["--accelerator", "rocm", "--target", "gfx1030"],
    command.replace(
      "python3 /probe.py",
      "printf gfx1030 >> /usr/lib/ollama/rocm/libggml-hip.so; python3 /probe.py",
    ),
  );
  assert.equal(hipTarget.status, 0, hipTarget.stderr);
});

test("CPU PID1 gate selects current Python minor and read-only environment without a shell or init process", async () => {
  const { pid1Command } = await import("./distribution-pid1-test.mjs");
  const args = pid1Command(
    {
      version: "3.13",
      site_packages: "/private/env/lib/python3.13/site-packages",
    },
    "/private/repo",
  );
  assert.deepEqual(args.slice(0, 6), [
    "run",
    "--rm",
    "--init=false",
    "--network",
    "none",
    "--mount",
  ]);
  assert.ok(args.includes("python:3.13-slim"));
  assert.ok(
    args.includes(
      "type=bind,source=/private/repo,target=/private/repo,readonly",
    ),
  );
  assert.ok(
    args.includes(
      "type=bind,source=/private/env/lib/python3.13/site-packages,target=/testdeps,readonly",
    ),
  );
  assert.deepEqual(
    args.slice(
      args.indexOf("python:3.13-slim") + 1,
      args.indexOf("python:3.13-slim") + 4,
    ),
    ["python", "-m", "pytest"],
  );
  assert.ok(
    args.includes("tests/test_ocr_initialize.py") &&
      args.includes("tests/test_ocr_provision.py"),
  );
  assert.throws(() =>
    pid1Command(
      { version: "latest", site_packages: "/private/env" },
      "/private/repo",
    ),
  );
  assert.match(
    readFileSync(".github/workflows/ci.yml", "utf8"),
    /make distribution-pid1-test/,
  );
});

test("qualification evidence cannot be reused for internally consistent replacement models, fixtures or devices", (t) => {
  for (const replace of [
    "model",
    "fixture",
    "fixture-table",
    "device",
    "report-model",
    "seal-model",
  ]) {
    const directory = temp(t),
      { manifest, bytes } = fixture(),
      native = manifest.native.rocm;
    const bundle = JSON.parse(bytes[native.bundle]),
      report = JSON.parse(bytes[native.qualification]),
      seal = JSON.parse(bytes[native.seal]);
    if (replace === "model") {
      const model = bundle.models.UVDoc;
      model.sha256 = hash("different-archive");
      model.files.UVDoc.sha256 = hash("different-weights");
      model.tree_sha256 = hash(
        JSON.stringify(
          Object.fromEntries(
            Object.entries(model.files)
              .sort()
              .map(([name, row]) => [name, row.sha256]),
          ),
        ),
      );
    }
    if (replace === "fixture" || replace === "fixture-table") {
      const row = bundle.fixtures.cases.find(
        (row) => row.kind === (replace === "fixture" ? "upright" : "table"),
      );
      row.sha256 = hash("different-upright");
      bundle.fixtures.files[row.path].sha256 = row.sha256;
      bundle.fixtures.tree_sha256 = hash(
        JSON.stringify(
          Object.fromEntries(
            Object.entries(bundle.fixtures.files)
              .sort()
              .map(([name, file]) => [name, file.sha256]),
          ),
        ),
      );
    }
    if (replace === "device") {
      report.device.name = "Different Radeon";
      report.device.gfx_target = "gfx1031";
      report.qualified_targets = ["gfx1031"];
    }
    if (replace === "report-model")
      report.model_content.UVDoc = hash("wrong-report-model");
    if (replace === "seal-model") {
      delete seal.model_content.UVDoc;
      report.manifest_sha256 = hash(JSON.stringify(seal));
    }
    for (const [path, value] of [
      [native.bundle, bundle],
      [native.qualification, report],
      [native.seal, seal],
    ]) {
      bytes[path] = JSON.stringify(value);
      manifest.assets[path] = {
        sha256: hash(bytes[path]),
        size_bytes: bytes[path].length,
      };
    }
    seed(directory, bytes);
    const output = join(directory, "out");
    assert.throws(
      () => writeBundle(manifest, "ocr-rocm", output, directory),
      /model|fixture|device|seal|qualification/i,
    );
    assert.ok(!readdirSync(directory).includes("out"));
  }
});

test("approved producer run is bound to exact workflow, maintainer actor, source and manual completion", async () => {
  const { validateInputRun, PRODUCER_WORKFLOW } =
    await import("./distribution-inputs.mjs");
  const request = {
    repository: "maintainer/vision-bill",
    sha: hash("source").slice(0, 40),
    version: "1.2.3",
    variant: "app",
  };
  const workflow = { id: 7, path: PRODUCER_WORKFLOW },
    permission = { role_name: "maintain", user: { login: "maintainer" } };
  const run = {
    id: 123,
    workflow_id: 7,
    path: PRODUCER_WORKFLOW,
    head_sha: request.sha,
    head_repository: { full_name: request.repository },
    event: "workflow_dispatch",
    conclusion: "success",
    actor: { login: "maintainer" },
    triggering_actor: { login: "maintainer" },
  };
  assert.equal(
    validateInputRun(run, request, workflow, permission),
    "maintainer",
  );
  for (const mutate of [
    (r) => (r.workflow_id = 8),
    (r) => (r.path = ".github/workflows/other.yml"),
    (r) => (r.actor.login = "foreign"),
    (r) => (r.triggering_actor.login = "foreign"),
    (r) => (r.head_sha = hash("old").slice(0, 40)),
    (r) => (r.event = "push"),
    (r) => (r.conclusion = "failure"),
  ]) {
    const bad = structuredClone(run);
    mutate(bad);
    assert.throws(() => validateInputRun(bad, request, workflow, permission));
  }
  assert.throws(() =>
    validateInputRun(run, request, workflow, {
      ...permission,
      role_name: "write",
    }),
  );
});

test("producer stages only approved exact-version inputs and consumer verifies actor-bound complete inventory", async (t) => {
  const { stageInputs, verifyStagedInputs } =
    await import("./distribution-inputs.mjs");
  const directory = temp(t),
    sourceRoot = join(directory, "approved"),
    request = {
      repository: "maintainer/vision-bill",
      sha: hash("source").slice(0, 40),
      version: "1.2.3",
      variant: "combined-rocm",
      run: "123",
    };
  const selected = join(
      sourceRoot,
      request.sha,
      request.version,
      request.variant,
    ),
    { manifest, bytes } = fixture();
  const plan = {
    schema_version: 1,
    source: manifest.source,
    version: request.version,
    platform: manifest.platform,
    postgres: manifest.images.postgres.reference,
    ollama_base: { accelerator: "rocm", reference: manifest.ollama.rocm.base },
    llm: manifest.llm,
    rocm: { targets: ["gfx1030"], render_node: "/dev/dri/renderD128" },
  };
  mkdirSync(selected, { recursive: true, mode: 0o700 });
  seed(selected, {
    "release-inputs.json": JSON.stringify(plan),
    "rocm/bundle.json": bytes["rocm/bundle.json"],
    "llm/LICENSE": bytes["llm/LICENSE"],
    "private-extra.txt": "must not upload",
  });
  const output = join(directory, "artifact");
  stageInputs(
    request,
    { actor: "maintainer", role: "maintain" },
    sourceRoot,
    output,
  );
  assert.ok(!readdirSync(output).includes("private-extra.txt"));
  verifyStagedInputs(output, request, { actor: "maintainer", run: "123" });
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "maintainer", run: "124" }),
  );
  writeFileSync(join(output, "unexpected.txt"), "must not consume");
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "maintainer", run: "123" }),
  );
  rmSync(join(output, "unexpected.txt"));
  writeFileSync(
    join(selected, "release-inputs.json"),
    JSON.stringify({ ...plan, version: "1.2.4" }),
  );
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "wrong-version"),
    ),
  );
  writeFileSync(join(selected, "release-inputs.json"), JSON.stringify(plan));
  const localBundle = JSON.parse(bytes["rocm/bundle.json"]);
  localBundle.models.UVDoc.source = "../../private-extra.txt";
  writeFileSync(
    join(selected, "rocm/bundle.json"),
    JSON.stringify(localBundle),
  );
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "traversal"),
    ),
  );
  localBundle.models.UVDoc.source = "model.tar";
  const localBytes = "synthetic local archive transport";
  localBundle.models.UVDoc.sha256 = hash(localBytes);
  localBundle.models.UVDoc.size_bytes = Buffer.byteLength(localBytes);
  writeFileSync(join(selected, "rocm/model.tar"), localBytes);
  writeFileSync(
    join(selected, "rocm/bundle.json"),
    JSON.stringify(localBundle),
  );
  const localOutput = join(directory, "local-artifact");
  stageInputs(
    request,
    { actor: "maintainer", role: "maintain" },
    sourceRoot,
    localOutput,
  );
  verifyStagedInputs(localOutput, request, { actor: "maintainer", run: "123" });
  assert.equal(
    readFileSync(join(localOutput, "rocm/model.tar"), "utf8"),
    localBytes,
  );
  writeFileSync(join(selected, "rocm/model.tar"), "changed archive");
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "wrong-archive"),
    ),
  );
  writeFileSync(join(selected, "rocm/bundle.json"), bytes["rocm/bundle.json"]);
  assert.throws(() =>
    verifyStagedInputs(
      output,
      { ...request, version: "1.2.4" },
      { actor: "maintainer", run: "123" },
    ),
  );
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "foreign", run: "123" }),
  );
  writeFileSync(join(output, "llm/LICENSE"), "changed");
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "maintainer", run: "123" }),
  );
  assert.throws(() =>
    stageInputs(
      { ...request, variant: "../escape" },
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "bad"),
    ),
  );
  const { symlinkSync, linkSync } = await import("node:fs");
  rmSync(join(selected, "llm/LICENSE"));
  writeFileSync(join(directory, "outside"), "private");
  linkSync(join(directory, "outside"), join(selected, "llm/LICENSE"));
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "hardlink"),
    ),
  );
  rmSync(join(selected, "llm/LICENSE"));
  symlinkSync(join(directory, "outside"), join(selected, "llm/LICENSE"));
  writeFileSync(join(directory, "outside"), "private");
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "symlink"),
    ),
  );
  const producer = readFileSync(
      ".github/workflows/distribution-inputs.yml",
      "utf8",
    ),
    consumer = readFileSync(
      ".github/workflows/distribution-release.yml",
      "utf8",
    );
  assert.match(producer, /workflow_dispatch:/);
  assert.match(producer, /actions\/upload-artifact@/);
  assert.match(producer, /name: distribution-inputs/);
  assert.match(consumer, /--check-run/);
  assert.match(consumer, /--verify/);
  assert.match(consumer, /DISTRIBUTION_INPUTS_ACTOR/);
});
