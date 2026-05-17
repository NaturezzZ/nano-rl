"""Ray cluster connect-or-create startup controller."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import warnings
from typing import Any, Mapping

from nano_rl.exceptions import RayClusterError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RayClusterStartupResult:
    """Auditable outcome of Ray runtime initialization."""

    startup_mode: str
    requested_address: str | None
    namespace: str | None
    created_local_cluster: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "startup_mode": self.startup_mode,
            "requested_address": self.requested_address,
            "namespace": self.namespace,
            "created_local_cluster": self.created_local_cluster,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class RayClusterController:
    """Own Ray runtime initialization before the actor graph is created.

    ``address=auto`` means connect to an existing Ray cluster first.  If that
    fails, v0.1 creates a single-machine local Ray cluster with the resolved
    role-scoped custom resources.  Explicit non-auto addresses keep fail-fast
    semantics so a typo or broken remote endpoint is not silently masked.
    """

    namespace: str | None
    ray_address: str | None
    node_custom_resources: Mapping[str, float]
    node_num_cpus: int | None = None
    dedup_logs: bool = True
    ray_module: Any | None = None

    def ensure_initialized(self) -> RayClusterStartupResult:
        self._configure_log_dedup_env()
        ray = self._ray_module()
        address = self._normalized_address()

        if ray.is_initialized():
            logger.info("Ray runtime already initialized: namespace=%s", self.namespace)
            return RayClusterStartupResult(
                startup_mode="already_initialized",
                requested_address=address,
                namespace=self.namespace,
                created_local_cluster=False,
            )

        if address == "auto":
            try:
                logger.info("Ray runtime: checking for existing Ray cluster")
                self._connect_existing(ray, address)
                logger.info("Ray runtime: using existing Ray cluster")
                return RayClusterStartupResult(
                    startup_mode="connected_existing",
                    requested_address=address,
                    namespace=self.namespace,
                    created_local_cluster=False,
                )
            except Exception as exc:
                if ray.is_initialized():
                    logger.info("Ray runtime: using existing Ray cluster")
                    return RayClusterStartupResult(
                        startup_mode="connected_existing",
                        requested_address=address,
                        namespace=self.namespace,
                        created_local_cluster=False,
                        fallback_reason=_format_exception(exc),
                    )
                fallback_reason = _format_exception(exc)
                logger.info("Ray runtime: no existing Ray cluster found; creating local Ray cluster")
                try:
                    self._create_local(ray)
                except Exception as create_exc:
                    logger.exception(
                        "failed to create local Ray cluster after auto connect failure: connect_error=%s",
                        fallback_reason,
                    )
                    raise RayClusterError(
                        "failed to connect to Ray cluster at address='auto' "
                        f"and failed to create a local Ray cluster; connect_error={fallback_reason}"
                    ) from create_exc
                logger.info("Ray runtime: local Ray cluster created")
                return RayClusterStartupResult(
                    startup_mode="created_local_after_auto_failed",
                    requested_address=address,
                    namespace=self.namespace,
                    created_local_cluster=True,
                    fallback_reason=fallback_reason,
                )

        if address in (None, "local"):
            try:
                logger.info("Ray runtime: creating local Ray cluster")
                self._create_local(ray)
            except Exception as exc:
                logger.exception("failed to create local Ray cluster")
                raise RayClusterError("failed to create a local Ray cluster") from exc
            logger.info("Ray runtime: local Ray cluster created")
            return RayClusterStartupResult(
                startup_mode="created_local",
                requested_address=address,
                namespace=self.namespace,
                created_local_cluster=True,
            )

        try:
            logger.info("connecting to Ray cluster: address=%s namespace=%s", address, self.namespace)
            self._connect_existing(ray, address)
        except Exception as exc:
            logger.exception("failed to connect to Ray cluster: address=%s", address)
            raise RayClusterError(f"failed to connect to Ray cluster at address={address!r}") from exc
        logger.info("connected to Ray cluster: address=%s namespace=%s", address, self.namespace)
        return RayClusterStartupResult(
            startup_mode="connected_existing",
            requested_address=address,
            namespace=self.namespace,
            created_local_cluster=False,
        )

    def _connect_existing(self, ray: Any, address: str) -> None:
        self._ray_init(ray, self._init_kwargs(address=address, resources=None))

    def _create_local(self, ray: Any) -> None:
        self._ray_init(
            ray,
            self._init_kwargs(
                address="local",
                resources=dict(self.node_custom_resources),
                num_cpus=self.node_num_cpus,
            )
        )

    def _init_kwargs(
        self,
        *,
        address: str | None,
        resources: dict[str, float] | None,
        num_cpus: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"namespace": self.namespace}
        if address is not None:
            kwargs["address"] = address
        if resources is not None:
            kwargs["resources"] = resources
        if num_cpus is not None:
            kwargs["num_cpus"] = num_cpus
        kwargs["logging_level"] = "warning"
        return {key: value for key, value in kwargs.items() if value is not None}

    def _normalized_address(self) -> str | None:
        if self.ray_address is None:
            return None
        address = self.ray_address.strip()
        return address or None

    def _ray_module(self) -> Any:
        if self.ray_module is not None:
            return self.ray_module
        try:
            import ray
        except ImportError as exc:
            raise RayClusterError("Ray is required to start the actor graph") from exc
        return ray

    def _ray_init(self, ray: Any, kwargs: dict[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Tip: In future versions of Ray, Ray will no longer override "
                    "accelerator visible devices env var.*"
                ),
                category=FutureWarning,
                module=r"ray\._private\.worker",
            )
            ray.init(**kwargs)

    def _configure_log_dedup_env(self) -> None:
        if not self.dedup_logs:
            os.environ["RAY_DEDUP_LOGS"] = "0"


def _format_exception(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__
