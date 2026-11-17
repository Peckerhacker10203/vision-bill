#!/usr/bin/env node
// CPU-only release assembly. No pulls, builds, inference or publication.
import { createHash } from "node:crypto";
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  lstatSync,
  realpathSync,
  renameSync,
  rmSync,
  mkdtempSync,
  existsSync,
  openSync,
  readSync,
  closeSync,
  copyFileSync,
} from "node:fs";
import { dirname, resolve, join, relative, sep } from "node:path";
import { pathToFileURL } from "node:url";

const schema = JSON.parse(
  readFileSync(
    new URL("../distribution/artifacts.schema.json", import.meta.url),
  ),
);
const variants = [
  "app",
  "ocr-cuda",
  "ocr-rocm",
  "combined-cuda",
  "combined-rocm",
];
const hash = (value) => createHash("sha256").update(value).digest("hex");
const canonical = (value) =>
  Array.isArray(value)
    ? value.map(canonical)
    : value && typeof value === "object"
      ? Object.fromEntries(
          Object.keys(value)
            .sort()
            .map((key) => [key, canonical(value[key])]),
        )
      : value;
const json = (value) => JSON.stringify(canonical(value), null, 2) + "\n";
const fail = (message) => {
  throw Error(message);
};
const check = (condition, message) => {
  if (!condition) fail(message);
};
function schemaCheck(value, rule, path = "manifest", root = schema) {
  if (rule.$ref)
    return schemaCheck(
      value,
      rule.$ref
        .split("/")
        .slice(1)
        .reduce((node, key) => node[key], root),
      path,
      root,
    );
  if ("const" in rule) check(value === rule.const, `${path}: invalid constant`);
  if (rule.enum) check(rule.enum.includes(value), `${path}: invalid choice`);
  if (rule.type === "object") {
    check(
      value !== null && typeof value === "object" && !Array.isArray(value),
      `${path}: expected object`,
    );
    check(
      Object.keys(value).length >= (rule.minProperties ?? 0),
      `${path}: incomplete object`,
    );
    check(
      Object.keys(value).length <= (rule.maxProperties ?? Infinity),
      `${path}: oversized object`,
    );
    for (const key of rule.required ?? [])
      check(Object.hasOwn(value, key), `${path}.${key}: required`);
    for (const [key, child] of Object.entries(value)) {
      const childRule = rule.properties?.[key] ?? rule.additionalProperties;
      check(childRule !== false, `${path}.${key}: unknown field`);
      if (typeof childRule === "object")
        schemaCheck(child, childRule, `${path}.${key}`, root);
    }
  } else if (rule.type === "string") {
    check(typeof value === "string", `${path}: expected string`);
    check(value.length >= (rule.minLength ?? 0), `${path}: empty string`);
    if (rule.pattern)
      check(new RegExp(rule.pattern).test(value), `${path}: invalid identity`);
  } else if (rule.type === "integer")
    check(
      Number.isSafeInteger(value) &&
        value >= (rule.minimum ?? 0) &&
        value > (rule.exclusiveMinimum ?? -Infinity) &&
        value <= (rule.maximum ?? Infinity),
      `${path}: invalid size`,
    );
  else if (rule.type === "boolean")
    check(typeof value === "boolean", `${path}: expected boolean`);
  else if (rule.type === "array") {
    check(
      Array.isArray(value) &&
        value.length >= (rule.minItems ?? 0) &&
        value.length <= (rule.maxItems ?? Infinity),
      `${path}: invalid array`,
    );
    for (const [index, child] of value.entries())
      schemaCheck(child, rule.items, `${path}[${index}]`, root);
  }
}
function safePath(path) {
  check(
    typeof path === "string" &&
      /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(path) &&
      path.split("/").every((part) => part && part !== "." && part !== ".."),
    "Unsafe asset path",
  );
  return path;
}
function rejectPlaceholders(value) {
  if (typeof value === "string") {
    check(
      !/(?:\b(?:example|placeholder|replace_me|todo|tbd|unexecuted|unresolved)\b|[<>]|\$\{)/i.test(
        value,
      ),
      "Placeholder release identity",
    );
    check(
      !/(?:^|@sha256:|^sha256:)([a-f0-9])\1{39,63}$/.test(value),
      "Placeholder digest",
    );
    if (value.startsWith("https://")) {
      const url = new URL(value);
      check(
        !url.username &&
          !url.password &&
          ![...url.searchParams.keys()].some((key) =>
            /token|secret|password|signature|api.?key/i.test(key),
          ),
        "Credential-bearing asset URL",
      );
    }
  } else if (value && typeof value === "object")
    for (const child of Object.values(value)) rejectPlaceholders(child);
}
export function validateArtifacts(manifest) {
  schemaCheck(manifest, schema);
  rejectPlaceholders(manifest);
  for (const path of Object.keys(manifest.assets)) safePath(path);
  const requireAsset = (path) => {
    safePath(path);
    check(Object.hasOwn(manifest.assets, path), `Missing asset: ${path}`);
  };
  requireAsset(manifest.verification.app_checks);
  if (manifest.llm) requireAsset(manifest.llm.license.file);
  for (const [accelerator, backend] of Object.entries(manifest.ollama ?? {})) {
    check(
      manifest.images[`ollama-${accelerator}`],
      "Missing accelerator-specific Ollama image",
    );
    requireAsset(backend.capability);
  }
  for (const [accelerator, native] of Object.entries(manifest.native ?? {})) {
    check(
      manifest.images[`ocr-${accelerator}`],
      "Missing native image identity",
    );
    for (const path of Object.values(native)) requireAsset(path);
    if (accelerator === "cuda")
      check(native.cuda_lock, "Missing resolved CUDA lock evidence");
  }
  return manifest;
}
const volume = (source, target, readOnly = false) => ({
  type: "volume",
  source,
  target,
  ...(readOnly ? { read_only: true } : {}),
});
const bind = (source, target) => ({
  type: "bind",
  source: `./release/${source}`,
  target,
  read_only: true,
  bind: { create_host_path: false },
});
const done = (name) => ({
  [name]: { condition: "service_completed_successfully" },
});
const device = (accelerator) =>
  accelerator === "cuda"
    ? {
        deploy: {
          resources: {
            reservations: {
              devices: [
                {
                  driver: "nvidia",
                  device_ids: ["${OCR_CUDA_DEVICE_ID:-0}"],
                  capabilities: ["gpu"],
                },
              ],
            },
          },
        },
      }
    : {
        devices: [
          "/dev/kfd:/dev/kfd",
          "${OCR_RENDER_NODE:-/dev/dri/renderD128}:/dev/dri/renderD128",
        ],
      };

export function renderBundle(manifest, variant) {
  validateArtifacts(manifest);
  check(variants.includes(variant), "Unknown distribution variant");
  const combined = variant.startsWith("combined-");
  const app = variant === "app" || combined;
  const accelerator = variant === "app" ? null : variant.split("-").at(-1);
  const names = [
    ...(app ? ["app", "postgres"] : []),
    ...(combined ? [`ollama-${accelerator}`] : []),
    ...(accelerator ? [`ocr-${accelerator}`] : []),
  ];
  for (const name of names)
    check(manifest.images[name], `Missing ${name} image`);
  if (combined) {
    check(manifest.llm, "Missing bundled model identity");
    check(
      manifest.ollama?.[accelerator],
      "Missing accelerator-specific Ollama backend evidence",
    );
  }
  if (accelerator)
    check(manifest.native?.[accelerator], "Missing native qualification");
  const services = {},
    volumes = {};
  const assets = new Set([manifest.verification.app_checks]);
  if (combined) assets.add(manifest.llm.license.file);
  if (combined) assets.add(manifest.ollama[accelerator].capability);
  function service(name, imageName, fields) {
    services[name] = {
      image: manifest.images[imageName].reference,
      platform: manifest.platform,
      ...fields,
    };
  }
  if (app) {
    for (const name of [
      "app-config",
      "pg-secret",
      "db-data",
      "uploads",
      "uploads-tmp",
      "logs",
    ])
      volumes[name] = {};
    const llm = combined
      ? [
          "--provider",
          "ollama",
          "--llm-host",
          "http://ollama:11434",
          "--model",
          manifest.llm.model,
          "--expected-model-digest",
          manifest.llm.digest,
        ]
      : [
          "--provider",
          "${LLM_PROVIDER:?Set LLM_PROVIDER}",
          "--llm-host",
          "${LLM_HOST:?Set LLM_HOST}",
          "--model",
          "${LLM_MODEL:?Set LLM_MODEL}",
        ];
    service("config-init", "app", {
      restart: "no",
      command: [
        "init",
        "--config-dir",
        "/app/config",
        "--postgres-password-file",
        "/run/pg-secret/password",
        ...llm,
      ],
      volumes: [
        volume("app-config", "/app/config"),
        volume("pg-secret", "/run/pg-secret"),
      ],
    });
    service("db", "postgres", {
      restart: "unless-stopped",
      depends_on: done("config-init"),
      environment: {
        POSTGRES_USER: "vision_bill",
        POSTGRES_DB: "vision_bill",
        POSTGRES_PASSWORD_FILE: "/run/pg-secret/password",
        PGDATA: "/var/lib/postgresql/data/pgdata",
      },
      volumes: [
        volume("db-data", "/var/lib/postgresql"),
        volume("pg-secret", "/run/pg-secret", true),
      ],
      healthcheck: {
        test: ["CMD-SHELL", "pg_isready -U vision_bill -d vision_bill"],
        interval: "5s",
        timeout: "5s",
        retries: 30,
      },
    });
    service("app", "app", {
      restart: "unless-stopped",
      depends_on: {
        ...done("config-init"),
        db: { condition: "service_healthy" },
      },
      environment: {
        VISION_BILL_CONFIG_PATH: "/app/config/config.yaml",
        OCR__ENDPOINT: combined ? "http://ocr:8082" : "${OCR_ENDPOINT:-}",
        OCR__TIMEOUT_SECONDS: "600",
        OCR__ALLOW_VISION_FALLBACK: "false",
      },
      ports: ["${APP_BIND_ADDRESS:-127.0.0.1}:${APP_PORT:-8080}:8080"],
      volumes: [
        volume("app-config", "/app/config"),
        volume("uploads", "/app/uploads"),
        volume("uploads-tmp", "/app/uploads_tmp"),
        volume("logs", "/app/logs"),
      ],
    });
  }
  if (accelerator) {
    volumes["ocr-state"] = {};
    volumes.coordination = {};
    const native = manifest.native[accelerator];
    for (const path of Object.values(native)) assets.add(path);
    const mounts = [
      volume("ocr-state", "/var/lib/ocr"),
      volume("coordination", "/coord"),
      bind(`${accelerator}`, "/release"),
    ];
    // Each accelerator's bundle owns one directory, including any local model archives.
    for (const path of Object.keys(manifest.assets))
      if (path.startsWith(`${accelerator}/`)) assets.add(path);
    check(
      native.bundle === `${accelerator}/bundle.json` &&
        native.image_config === `${accelerator}/image-config.json`,
      "Native release paths must match initializer mounts",
    );
    const lock = {
      OCR_INSTALLATION_LOCK_PATH: "/coord/installation.lock",
      OCR_STATE_DIR: "/var/lib/ocr",
    };
    service("ocr-init", `ocr-${accelerator}`, {
      init: false,
      restart: "no",
      entrypoint: ["python", "-m", "ocr_service.initialize"],
      command: [
        "--bundle",
        "/release/bundle.json",
        "--state-dir",
        "/var/lib/ocr",
        "--image-config",
        "/release/image-config.json",
        "--image-id",
        manifest.images[`ocr-${accelerator}`].config_digest,
      ],
      environment: lock,
      volumes: mounts,
      ...device(accelerator),
    });
    service("ocr", `ocr-${accelerator}`, {
      init: false,
      restart: "unless-stopped",
      command: ["python", "-m", "ocr_service.entrypoint"],
      depends_on: done("ocr-init"),
      environment: {
        ...lock,
        OCR_SERVICE_MODE: combined
          ? "managed"
          : "${OCR_SERVICE_MODE:-standalone}",
      },
      volumes: mounts,
      ...(!combined
        ? { ports: ["${OCR_BIND_ADDRESS:-127.0.0.1}:${OCR_PORT:-8082}:8082"] }
        : {}),
      ...device(accelerator),
    });
    if (combined) {
      volumes["ollama-data"] = {};
      service("ollama", `ollama-${accelerator}`, {
        init: false,
        restart: "unless-stopped",
        command: ["serve"],
        depends_on: done("ocr-init"),
        environment: {
          OLLAMA_HOST: "0.0.0.0:11434",
          OCR_INSTALLATION_LOCK_PATH: "/coord/installation.lock",
        },
        volumes: [
          volume("ollama-data", "/root/.ollama"),
          volume("coordination", "/coord"),
        ],
        ...device(accelerator),
      });
      service("model-init", "app", {
        restart: "on-failure",
        command: [
          "provision-model",
          "--config-dir",
          "/app/config",
          "--expected-digest",
          manifest.llm.digest,
          "--bundled-model",
          manifest.llm.model,
        ],
        depends_on: {
          ...done("config-init"),
          ollama: { condition: "service_started" },
        },
        volumes: [volume("app-config", "/app/config")],
      });
      services.app.depends_on["model-init"] = {
        condition: "service_completed_successfully",
      };
    }
  }
  const metadata = {
    schema_version: 1,
    version: manifest.version,
    source: manifest.source,
    platform: manifest.platform,
    variant,
    images: Object.fromEntries(
      names.map((name) => [name, manifest.images[name]]),
    ),
    ...(combined ? { llm: manifest.llm } : {}),
    ...(combined
      ? { ollama: { [accelerator]: manifest.ollama[accelerator] } }
      : {}),
    verification: manifest.verification,
    ...(accelerator
      ? { native: { [accelerator]: manifest.native[accelerator] } }
      : {}),
    assets: Object.fromEntries(
      [...assets].sort().map((path) => [path, manifest.assets[path]]),
    ),
  };
  return {
    files: {
      "compose.yaml": json({ services, volumes }),
      "release.json": json(metadata),
    },
    assets: [...assets].sort(),
  };
}
export function fileIdentity(path) {
  const descriptor = openSync(path, "r"),
    digest = createHash("sha256"),
    buffer = Buffer.allocUnsafe(1024 * 1024);
  let size = 0;
  try {
    let count;
    while ((count = readSync(descriptor, buffer, 0, buffer.length, null)) > 0) {
      digest.update(buffer.subarray(0, count));
      size += count;
    }
  } finally {
    closeSync(descriptor);
  }
  return { sha256: digest.digest("hex"), size_bytes: size };
}
function checkedRead(base, path, record) {
  safePath(path);
  const root = realpathSync(base),
    full = resolve(root, path);
  check(
    !relative(root, realpathSync(full)).startsWith(`..${sep}`),
    "Asset outside manifest directory",
  );
  let cursor = full;
  while (cursor !== root) {
    check(!lstatSync(cursor).isSymbolicLink(), "Symlink release asset");
    cursor = dirname(cursor);
  }
  check(lstatSync(full).isFile(), "Nonregular release asset");
  const actual = fileIdentity(full);
  check(
    actual.size_bytes === record.size_bytes && actual.sha256 === record.sha256,
    `Asset hash identity mismatch: ${path}`,
  );
  return full;
}
function verifyEvidence(manifest, variant, contents) {
  const read = (path) => {
    check(
      manifest.assets[path].size_bytes <= 32 * 1024 * 1024,
      "Metadata exceeds size bound",
    );
    return JSON.parse(readFileSync(contents[path]));
  };
  const app = read(manifest.verification.app_checks);
  check(
    app.passed === true &&
      app.source_commit === manifest.source.commit &&
      ["backend", "frontend", "e2e-backend", "e2e-frontend"].every((name) =>
        app.checks?.includes(name),
      ),
    "App source checks are incomplete",
  );
  if (variant === "app") return;
  const accelerator = variant.split("-").at(-1),
    native = manifest.native[accelerator],
    image = manifest.images[`ocr-${accelerator}`];
  if (variant.startsWith("combined")) {
    const expected = manifest.images[`ollama-${accelerator}`],
      backend = manifest.ollama[accelerator],
      proof = read(backend.capability);
    check(
      proof.scope === "backend-package-only" &&
        proof.accelerator === accelerator &&
        proof.base_reference === backend.base &&
        proof.image_identity === expected.reference &&
        proof.config_digest === expected.config_digest &&
        proof.dependencies_resolved === true,
      "Ollama accelerator/image capability identity mismatch",
    );
    const libraries = Object.entries(proof.backend_libraries ?? {}),
      library = accelerator === "cuda" ? "cuda" : "hip";
    check(
      libraries.length > 0 &&
        libraries.every(
          ([path, digest]) =>
            new RegExp(
              `^/usr/lib/ollama/.*/?libggml-${library}\\.so(?:\\.[0-9]+)*$`,
            ).test(path) && /^[a-f0-9]{64}$/.test(digest),
        ),
      "Selected Ollama native backend library evidence missing",
    );
  }
  const bundle = read(native.bundle),
    config = read(native.image_config),
    report = read(native.qualification);
  const nativeSchema = JSON.parse(
    readFileSync(
      new URL(
        "../paddleocr-service/release-bundle.schema.json",
        import.meta.url,
      ),
    ),
  );
  schemaCheck(bundle, nativeSchema, "native bundle", nativeSchema);
  check(
    JSON.stringify(Object.keys(bundle.models).sort()) ===
      JSON.stringify(
        [
          "PP-DocLayoutV3",
          "PP-LCNet_x1_0_doc_ori",
          "PaddleOCR-VL-1.6-0.9B",
          "UVDoc",
        ].sort(),
      ),
    "Exactly four approved model identities required",
  );
  check(
    bundle.expected_packages.paddleocr === "3.7.0" &&
      bundle.expected_packages.paddlex === "3.7.2",
    "Pinned OCR packages required",
  );
  const paddle = Object.keys(bundle.expected_packages).filter((name) =>
    ["paddlepaddle", "paddlepaddle-gpu"].includes(name),
  );
  check(
    paddle.length === 1 &&
      bundle.expected_packages[paddle[0]].split("+")[0] === "3.3.1",
    "Pinned native Paddle required",
  );
  check(
    bundle.source_files["profile.json"] === bundle.profile_sha256 &&
      bundle.source_files["runtime-lock.json"],
    "Source profile/lock identities missing",
  );
  for (const [name, archive] of Object.entries({
    ...bundle.models,
    fixtures: bundle.fixtures,
  })) {
    for (const path of Object.keys(archive.files)) {
      safePath(path);
      check(
        !path
          .split("/")
          .slice(0, -1)
          .some(
            (_, i, parts) => archive.files[parts.slice(0, i + 1).join("/")],
          ),
        "Archive path collision",
      );
    }
    check(
      archive.files[safePath(archive.license.text_file)]?.size_bytes > 0,
      "Sealed license text missing",
    );
    check(
      Object.values(archive.files).reduce(
        (sum, row) => sum + row.size_bytes,
        0,
      ) === archive.unpacked_bytes,
      "Complete unpacked inventory missing",
    );
    const tree = Object.fromEntries(
      Object.entries(archive.files)
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([path, row]) => [path, row.sha256]),
    );
    check(
      hash(JSON.stringify(tree)) === archive.tree_sha256,
      "Complete archive tree hash mismatch",
    );
    if (/^https:\/\//.test(archive.source))
      check(
        new URL(archive.source).protocol === "https:",
        "HTTPS asset required",
      );
    else {
      const path = `${accelerator}/${safePath(archive.source)}`;
      check(
        contents[path] &&
          manifest.assets[path].sha256 === archive.sha256 &&
          manifest.assets[path].size_bytes === archive.size_bytes,
        `Missing archive identity: ${name}`,
      );
    }
  }
  check(
    new Set(bundle.fixtures.cases.map((row) => row.kind)).size === 5 &&
      new Set(bundle.fixtures.cases.map((row) => row.sha256)).size === 5,
    "Five distinct fixture kinds/identities required",
  );
  for (const row of bundle.fixtures.cases) {
    check(
      bundle.fixtures.files[safePath(row.path)]?.sha256 === row.sha256 &&
        bundle.fixtures.files[safePath(row.license.text_file)]?.size_bytes > 0,
      "Fixture inventory/license mismatch",
    );
    if (["table", "formula"].includes(row.kind))
      check(
        row.required_labels?.includes(row.kind),
        "Native structure qualification label missing",
      );
  }
  check(
    config.os === "linux" && config.architecture === "amd64",
    "Wrong OCI platform",
  );
  check(
    ["", "0", "root", "0:0", "root:root"].includes(config.config?.User ?? ""),
    "Prepared native image must run as root for selected device owner access",
  );
  check(
    `sha256:${manifest.assets[native.image_config].sha256}` ===
      image.config_digest &&
      bundle.image_config_sha256 === image.config_digest.slice(7),
    "Raw image config digest identity mismatch",
  );
  check(
    bundle.accelerator === accelerator &&
      bundle.image_identity === image.reference,
    "Native bundle image identity mismatch",
  );
  check(
    report.qualified === true &&
      report.stage === "complete" &&
      report.accelerator === accelerator &&
      report.image_identity === image.reference,
    "Native hardware qualification incomplete",
  );
  const seal = read(native.seal);
  const modelContent = Object.fromEntries(
    Object.entries(bundle.models).map(([name, model]) => [
      name,
      model.tree_sha256,
    ]),
  );
  check(
    json(seal.model_content) === json(modelContent) &&
      json(report.model_content) === json(modelContent),
    "Qualified report/seal model inventory differs from distributed models",
  );
  check(
    json(seal.qualified_target) ===
      json(
        Object.fromEntries(
          Object.entries(report.device ?? {}).filter(
            ([key]) => key !== "compute_probe",
          ),
        ),
      ),
    "Qualified seal device identity differs from report",
  );
  check(
    seal.smoke_sha256 ===
      bundle.fixtures.cases.find((row) => row.kind === "upright").sha256,
    "Qualified seal smoke fixture differs from distributed fixture",
  );
  check(
    report.manifest_sha256 === manifest.assets[native.seal].sha256 &&
      ["profile_id", "runtime_fingerprint"].every((key) =>
        /^[a-f0-9]{64}$/.test(report[key]),
      ) &&
      seal.image_identity === image.reference &&
      seal.paddle_identity === `paddle:${bundle.expected_paddle_commit}`,
    "Native qualification evidence seal mismatch",
  );
  const target = report.device?.gfx_target || report.device?.name;
  if (variant === "combined-rocm") {
    const proof = read(manifest.ollama.rocm.capability);
    check(
      proof.target_markers?.includes(target) &&
        Object.values(proof.target_files ?? {}).length > 0 &&
        Object.entries(proof.target_files).every(
          ([path, digest]) =>
            /^\/usr\/lib\/ollama\/.*\/?libggml-hip\.so(?:\.[0-9]+)*$/.test(
              path,
            ) && /^[a-f0-9]{64}$/.test(digest),
        ),
      "Actual AMD target absent from Ollama package evidence",
    );
  }
  check(
    target && report.qualified_targets?.includes(target),
    "Actual qualified device target missing",
  );
  if (accelerator === "rocm")
    check(
      /^gfx[0-9a-f]+$/.test(report.device?.gfx_target),
      "Actual AMD GFX target evidence missing",
    );
  check(
    report.cases?.length === 10 &&
      ["upright", "rotated", "warped", "table", "formula"].every((kind) =>
        [0, 1].every(
          (worker) =>
            report.cases.filter(
              (row) =>
                row.kind === kind &&
                row.fresh_worker === worker &&
                row.source_sha256 ===
                  bundle.fixtures.cases.find((fixture) => fixture.kind === kind)
                    .sha256 &&
                row.passed === true,
            ).length === 1,
        ),
      ),
    "Full native fresh-worker qualification missing",
  );
  if (accelerator === "cuda") {
    const lock = read(native.cuda_lock);
    check(
      lock.index ===
        "https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/" &&
        ["www.paddlepaddle.org.cn", "paddle-whl.bj.bcebos.com"].includes(
          new URL(lock.wheel_url).hostname,
        ) &&
        new URL(lock.wheel_url).protocol === "https:",
      "Official CUDA host evidence missing",
    );
    check(
      lock.metadata_name === "paddlepaddle-gpu" &&
        lock.metadata_version === "3.3.1" &&
        lock.platform === "linux-x86_64-cp312" &&
        Array.isArray(lock.requires_dist) &&
        lock.requires_dist.length > 0 &&
        [
          "index_sha256",
          "wheel_sha256",
          "metadata_sha256",
          "lock_sha256",
        ].every((key) => /^[a-f0-9]{64}$/.test(lock[key])),
      "Official CUDA wheel/metadata resolution evidence missing",
    );
    check(
      contents["cuda/requirements.cuda.lock"] &&
        manifest.assets["cuda/requirements.cuda.lock"].sha256 ===
          lock.lock_sha256,
      "Resolved CUDA lock hash mismatch",
    );
  }
  // Native initializer independently enforces the complete release-bundle schema,
  // source locks, model inventories/licenses and fresh hardware qualification.
  rejectPlaceholders(bundle);
  rejectPlaceholders(report);
}
export function writeBundle(manifest, variant, output, base) {
  const rendered = renderBundle(manifest, variant),
    contents = {};
  for (const path of rendered.assets)
    contents[path] = checkedRead(base, path, manifest.assets[path]);
  verifyEvidence(manifest, variant, contents);
  check(
    !existsSync(output),
    "Output already exists; immutable bundles cannot be overwritten",
  );
  mkdirSync(dirname(resolve(output)), { recursive: true });
  const temporary = mkdtempSync(
    join(dirname(resolve(output)), ".distribution-"),
  );
  try {
    const identities = {};
    for (const [path, bytes] of Object.entries(rendered.files)) {
      writeFileSync(join(temporary, path), bytes, { mode: 0o644, flag: "wx" });
      identities[path] = hash(bytes);
    }
    for (const [path, source] of Object.entries(contents)) {
      const destination = join(temporary, "release", path);
      mkdirSync(dirname(destination), { recursive: true });
      copyFileSync(source, destination);
      identities[`release/${path}`] = manifest.assets[path].sha256;
    }
    writeFileSync(
      join(temporary, "SHA256SUMS"),
      Object.keys(identities)
        .sort()
        .map((path) => `${identities[path]}  ${path}\n`)
        .join(""),
      { mode: 0o644, flag: "wx" },
    );
    renameSync(temporary, output);
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
}
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  try {
    const args = process.argv.slice(2),
      options = {};
    check(
      args.length === 6,
      "Usage: --manifest FILE --variant VARIANT --output DIR",
    );
    for (let i = 0; i < args.length; i += 2) {
      check(
        ["--manifest", "--variant", "--output"].includes(args[i]) &&
          !options[args[i]],
        "Invalid CLI option",
      );
      options[args[i]] = args[i + 1];
    }
    const path = resolve(options["--manifest"]);
    writeBundle(
      JSON.parse(readFileSync(path)),
      options["--variant"],
      resolve(options["--output"]),
      dirname(path),
    );
  } catch (error) {
    process.stderr.write(`Distribution failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}
