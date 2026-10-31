"""Synthetic IT/cloud incident generator for demos and tests.

IMPORTANT: this is *simulated* data. Real tickets will be messier. It exists so the
whole pipeline can be demonstrated end-to-end. Replace `data/tickets.csv` with an
export from ServiceNow/Jira (same columns) to run on real data.

Severity is derived from: incident type (base impact) + environment + service tier
+ customer-facing + users affected + random noise (`score_noise` = context that is NOT
visible in the ticket; raise it to make the problem harder / more realistic), then a share of labels are
flipped by one level to mimic inconsistent human triage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (name, tier 1=critical..3=low, customer_facing, introduced_on)
SERVICES = [
    ("checkout-service", 1, True, "2025-01-01"),
    ("payments-api", 1, True, "2025-01-01"),
    ("auth-service", 1, True, "2025-01-01"),
    ("customer-portal", 1, True, "2025-01-01"),
    ("order-management", 2, True, "2025-01-01"),
    ("search-service", 2, True, "2025-01-01"),
    ("notification-service", 2, True, "2025-01-01"),
    ("inventory-sync", 2, False, "2025-01-01"),
    ("data-warehouse", 2, False, "2025-01-01"),
    ("reporting-dashboard", 3, False, "2025-01-01"),
    ("internal-wiki", 3, False, "2025-01-01"),
    ("hr-portal", 3, False, "2025-01-01"),
    ("ml-inference-gateway", 1, True, "2026-03-01"),   # new service -> simulates drift
    ("feature-store", 2, False, "2026-03-01"),         # new service -> simulates drift
]
OWNER = {1: "Platform-OnCall", 2: "App-Support", 3: "IT-Helpdesk"}

# (category, subcategory, nature, base_impact 0-10, [short description templates])
CATALOG = [
    ("IAM", "Leaked access keys", "security", 9.0, ["AWS access key found in public GitHub repo for {svc}", "Leaked access keys detected for {svc} service account", "Credentials exposed in commit history, {svc} IAM key compromised"]),
    ("IAM", "Root account login", "security", 9.0, ["Root account login detected without MFA", "Unexpected root user sign-in from unknown IP", "Root account used to modify IAM policies"]),
    ("IAM", "SSO outage", "availability", 8.5, ["SSO login failing for all users", "Okta SSO outage, users cannot sign in to {svc}", "Identity provider unreachable, authentication errors on {svc}"]),
    ("IAM", "Over-permissive policy", "hygiene", 4.0, ["IAM policy with wildcard permissions attached to {svc} role", "Audit flagged admin * policy on {svc} service role"]),
    ("IAM", "Stale access keys", "hygiene", 1.8, ["Access keys older than 180 days found for {svc}", "Stale IAM keys not rotated for {svc}", "Orphaned user accounts after offboarding in {svc} account"]),
    ("IAM", "Role assumption failure", "availability", 5.0, ["AssumeRole AccessDenied for {svc} pipeline", "Service account token expired, {svc} jobs failing auth"]),
    ("Threats", "Ransomware", "security", 10.0, ["Ransomware detected on instances serving {svc}", "Encrypted files and ransom note found on {svc} file server"]),
    ("Threats", "Crypto mining", "security", 8.0, ["Crypto mining activity detected on {svc} instances", "GuardDuty CryptoCurrency mining finding on {svc} hosts"]),
    ("Threats", "DDoS attack", "availability", 8.5, ["DDoS attack hitting {svc} endpoints", "Massive traffic spike, suspected DDoS on {svc}"]),
    ("Threats", "Data exfiltration", "security", 9.2, ["Unusual outbound data transfer from {svc} database host", "Possible data exfiltration, large egress from {svc}"]),
    ("Threats", "Vulnerability alert", "security", 4.5, ["Critical CVE reported in base image used by {svc}", "Vulnerability scan flagged unpatched library in {svc}"]),
    ("Data exposure", "Public bucket", "security", 8.0, ["Storage bucket with customer data is publicly accessible", "Public storage bucket exposes files for {svc}"]),
    ("Data exposure", "Secrets in code", "security", 5.5, ["Secrets found in logs of {svc}", "Hardcoded password detected in {svc} repository"]),
    ("Audit", "Logging disabled", "security", 6.5, ["CloudTrail logging disabled in production account", "Audit logs stopped flowing for {svc}"]),
    ("Network", "VPN down", "availability", 6.0, ["Site-to-site VPN tunnel down", "VPN connectivity lost to on-prem network"]),
    ("Network", "DNS failure", "availability", 8.0, ["DNS resolution failing for {svc}", "Route53 records missing, {svc} hostname not resolving"]),
    ("Network", "NAT gateway failure", "availability", 7.0, ["NAT gateway failure, private subnets lost internet access", "Outbound connections from {svc} timing out, NAT gateway issue"]),
    ("Network", "Firewall rule blocking", "availability", 5.0, ["Security group change blocking traffic to {svc}", "New firewall rule dropping legitimate requests to {svc}"]),
    ("Network", "IP exhaustion", "capacity", 5.0, ["Subnet running out of IP addresses for {svc}", "Cannot launch new pods, subnet IP exhaustion"]),
    ("Network", "Latency or packet loss", "performance", 5.0, ["High latency and packet loss between regions affecting {svc}", "Network latency spike on {svc} traffic"]),
    ("Network", "Certificate expired", "availability", 7.5, ["TLS certificate expired on {svc} endpoint", "SSL certificate expiry causing handshake failures on {svc}"]),
    ("Network", "CDN cache issue", "performance", 2.5, ["CDN serving stale content for {svc}", "Cache invalidation not working on CDN for {svc}"]),
    ("Load Balancer", "All targets unhealthy", "availability", 9.0, ["All targets unhealthy behind load balancer for {svc}", "ALB reports zero healthy hosts for {svc}"]),
    ("Load Balancer", "5xx spike", "availability", 7.0, ["Spike of 502/503 errors on load balancer for {svc}", "5xx error rate above threshold on {svc} ALB"]),
    ("Load Balancer", "Listener misconfiguration", "availability", 5.0, ["Load balancer listener rule misrouting traffic for {svc}", "Wrong path routing rule on ALB for {svc}"]),
    ("Load Balancer", "WAF false positive", "availability", 4.5, ["WAF blocking legitimate users of {svc}", "WAF rule false positive rejecting valid requests to {svc}"]),
    ("Load Balancer", "Idle timeout drops", "performance", 3.5, ["Connections dropped by load balancer idle timeout on {svc}", "Intermittent connection resets via LB for {svc}"]),
    ("Compute", "Instance failure", "availability", 4.5, ["EC2 instance status check failed for {svc}", "VM unreachable, failed health check on {svc} node"]),
    ("Compute", "Resource saturation", "performance", 4.2, ["High CPU utilisation on {svc} hosts", "Disk usage at 95 percent on {svc} server", "Memory pressure on {svc} instances"]),
    ("Kubernetes", "Pod CrashLoopBackOff", "availability", 5.2, ["Pods in CrashLoopBackOff for {svc}", "{svc} deployment pods crash looping after restart"]),
    ("Kubernetes", "Node NotReady", "availability", 5.8, ["Kubernetes nodes NotReady in {svc} cluster", "Worker node NotReady, pods being evicted for {svc}"]),
    ("Kubernetes", "OOMKilled", "performance", 4.5, ["Containers OOMKilled in {svc}", "{svc} pods repeatedly OOMKilled"]),
    ("Kubernetes", "Control plane unavailable", "availability", 8.8, ["Kubernetes API server unreachable", "Cluster control plane down, cannot deploy {svc}"]),
    ("Serverless", "Function timeouts", "performance", 5.0, ["Lambda timeouts increasing for {svc}", "Function throttling and timeouts on {svc}"]),
    ("Compute", "Autoscaling failure", "capacity", 6.0, ["Autoscaling group failed to scale out for {svc}", "Capacity not available in zone, {svc} cannot scale"]),
    ("Compute", "Spot interruption", "availability", 2.2, ["Spot instances reclaimed for {svc} batch workers", "Preemptible VM interruption on {svc} jobs"]),
    ("Compute", "Patch failure", "hygiene", 3.2, ["OS patching failed on {svc} hosts", "Bad AMI used in {svc} launch template"]),
    ("Database", "Primary database down", "availability", 10.0, ["Primary database is down for {svc}", "Primary RDS instance unreachable, {svc} cannot connect", "Production database crashed for {svc}"]),
    ("Database", "Replication lag", "performance", 6.0, ["Replication lag growing on {svc} read replica", "Replica lag over 10 minutes for {svc}"]),
    ("Database", "Connection exhaustion", "availability", 7.0, ["Too many connections, connection pool exhausted for {svc}", "DB max connections reached by {svc}"]),
    ("Database", "Slow queries", "performance", 4.0, ["Slow queries degrading {svc} database", "Long running queries and lock waits on {svc} DB"]),
    ("Database", "Storage full", "capacity", 7.5, ["Database storage almost full for {svc}", "Disk full on {svc} database volume"]),
    ("Database", "Backup failure", "reliability", 4.5, ["Nightly database backup failed for {svc}", "Snapshot backup job failed on {svc} DB"]),
    ("Database", "Failed migration", "availability", 6.0, ["Schema migration failed halfway on {svc}", "DB migration error blocking release of {svc}"]),
    ("Database", "Cache outage", "availability", 6.5, ["Redis cluster unavailable for {svc}", "Cache node down, evictions and stampede on {svc}"]),
    ("Database", "Data deleted or corrupted", "data_loss", 9.5, ["Table dropped by mistake in {svc} database", "Data corruption detected in {svc} database"]),
    ("Storage", "Object storage errors", "availability", 5.0, ["Elevated error rate on object storage for {svc}", "Object storage request throttling affecting {svc}"]),
    ("Storage", "Volume corruption", "data_loss", 7.0, ["Volume corruption detected on {svc} disk", "Block volume detached unexpectedly from {svc}"]),
    ("Storage", "Accidental deletion", "data_loss", 8.0, ["Objects deleted by lifecycle policy for {svc}", "Accidental deletion of files in {svc} bucket"]),
    ("Storage", "Replication failure", "reliability", 4.5, ["Cross-region replication failing for {svc} bucket", "Storage replication lag for {svc}"]),
    ("Storage", "Mount failure", "availability", 5.0, ["NFS mount failure on {svc} hosts", "File share latency and mount errors for {svc}"]),
    ("Application", "Bad deployment", "availability", 7.5, ["Release caused errors in {svc}, rollback needed", "Bad deployment of {svc} increasing error rate"]),
    ("Application", "Memory leak", "performance", 5.0, ["Memory leak causing restarts of {svc}", "{svc} application crashes with out of memory"]),
    ("Application", "Downstream timeouts", "performance", 6.0, ["Timeouts calling downstream API from {svc}", "Third-party API errors impacting {svc}"]),
    ("Application", "Payment gateway down", "availability", 9.0, ["Payment gateway not responding for {svc}", "Payment provider outage, transactions failing in {svc}"]),
    ("Application", "Queue backlog", "performance", 5.5, ["Message queue backlog growing for {svc}", "Kafka consumer lag high on {svc}"]),
    ("Application", "Feature flag misconfiguration", "availability", 6.0, ["Wrong feature flag enabled for {svc}", "Config change broke {svc} behaviour"]),
    ("Application", "Batch job failure", "reliability", 3.2, ["Scheduled job failed for {svc}", "Nightly batch job failure in {svc}", "ETL pipeline failed, stale reports in {svc}"]),
    ("CI/CD", "Pipeline failure", "reliability", 2.5, ["CI pipeline failing for {svc}", "Build agent unavailable for {svc} pipeline"]),
    ("CI/CD", "Artifact repository down", "availability", 4.0, ["Artifact repository unreachable, builds blocked", "Container registry pull failures for {svc}"]),
    ("CI/CD", "IaC apply failure", "reliability", 4.0, ["Terraform apply failed for {svc}", "State lock stuck on Terraform for {svc}"]),
    ("CI/CD", "Destructive IaC change", "data_loss", 9.0, ["Terraform change destroyed production resources of {svc}", "IaC run deleted {svc} infrastructure"]),
    ("Observability", "Monitoring agent down", "hygiene", 3.0, ["Monitoring agent not reporting from {svc} hosts", "Metrics gap on {svc} dashboards"]),
    ("Observability", "Alert storm", "hygiene", 3.0, ["Alert storm flooding on-call channel", "Noisy duplicate alerts for {svc}"]),
    ("Observability", "Log ingestion delay", "performance", 3.0, ["Log ingestion delayed for {svc}", "Logging pipeline lag affecting {svc} troubleshooting"]),
    ("Cloud Provider", "Region outage", "availability", 10.0, ["Cloud region outage affecting {svc}", "Provider reports regional service disruption, {svc} down"]),
    ("Cloud Provider", "AZ degradation", "availability", 8.0, ["Availability zone degradation, {svc} instances impaired", "AZ network issues impacting {svc}"]),
    ("Cloud Provider", "Control plane API errors", "availability", 7.0, ["Cloud control plane API errors, cannot modify resources", "Console and API throttling for {svc}"]),
    ("Cloud Provider", "Quota reached", "capacity", 4.5, ["Service quota limit reached for {svc}", "Cannot create resources, vCPU quota exceeded for {svc}"]),
    ("Cloud Provider", "Planned maintenance", "hygiene", 2.8, ["Provider maintenance event scheduled for {svc} hosts", "Upcoming forced reboot maintenance on {svc} instances"]),
    ("Disaster Recovery", "DR failover failure", "reliability", 5.5, ["DR failover test failed for {svc}", "RPO breached, cross-region replication behind for {svc}"]),
    ("Compliance", "Policy violation", "hygiene", 2.0, ["Resource tagging violations in {svc} account", "CIS benchmark failures reported for {svc}", "Unapproved service in use by {svc} team"]),
    ("Compliance", "Data residency violation", "hygiene", 6.0, ["Data stored in wrong region for {svc}", "PCI audit finding on {svc} environment"]),
    ("Cost", "Cost anomaly", "hygiene", 4.2, ["Unexpected cost spike in {svc} account", "Budget alert triggered for {svc}"]),
    ("Cost", "Orphaned resources", "hygiene", 1.2, ["Orphaned volumes and idle IPs in {svc}", "Unattached disks found in {svc} account"]),
    ("AI/ML", "Model endpoint failure", "availability", 6.5, ["Model serving endpoint failing for {svc}", "Inference latency spike on {svc}"]),
    ("AI/ML", "Training failure", "reliability", 2.5, ["Training job failed for {svc}", "GPU quota unavailable for {svc} experiments"]),
]

DETAILS = {
    "availability": ["Service is not responding as expected.", "Health checks are failing and error rates are elevated.", "Requests are failing or timing out."],
    "performance": ["Response times have degraded noticeably.", "Metrics show sustained saturation over the last hour.", "Users report slowness."],
    "security": ["Flagged by the security tooling and needs review.", "Activity does not match expected baseline behaviour.", "Security team has been notified."],
    "data_loss": ["Data may be missing or unreadable.", "Recovery options are being evaluated.", "Integrity of stored data is in question."],
    "capacity": ["A resource limit has been reached.", "New resources cannot be provisioned right now.", "Usage has been growing steadily."],
    "reliability": ["The job or process did not complete successfully.", "Last successful run was more than a day ago.", "Retries have not helped."],
    "hygiene": ["Found during routine review.", "No immediate service impact observed.", "Needs cleanup or follow up."],
}
ENV_PHRASES = {
    "prod": ["Environment: production.", "Observed in prod.", "This is the live production environment.", ""],
    "staging": ["Environment: staging.", "Seen in the staging environment.", ""],
    "dev": ["Environment: dev.", "Seen in the development environment.", "Only reproducible in dev.", ""],
}
IMPACT = {
    "high": ["Customers are unable to complete transactions.", "Widespread impact reported across multiple regions.", "All users affected, no workaround available.", "Business critical service impacted."],
    "medium": ["Some users are seeing intermittent errors.", "Partial degradation, workaround exists for most users.", "A subset of customers is affected."],
    "low": ["No customer impact reported.", "Only internal users affected.", "Minor inconvenience, low priority.", "No user impact at this time."],
    "none": ["", "", "Please investigate."],
}
ALERT_SRC = ["CloudWatch ALARM", "Azure Monitor", "Datadog", "Prometheus", "PagerDuty"]
USER_PREFIX = ["Hi team,", "Hello,", "Urgent:", "FYI -", "Please help.", ""]

# score -> severity cut points (tuned to give roughly P1 5%, P2 15%, P3 35%, P4 45%)
CUT_P1, CUT_P2, CUT_P3 = 10.2, 7.6, 4.4


def _severity_from_score(score: np.ndarray) -> np.ndarray:
    sev = np.full(score.shape, 3)          # P4
    sev[score >= CUT_P3] = 2               # P3
    sev[score >= CUT_P2] = 1               # P2
    sev[score >= CUT_P1] = 0               # P1
    return sev


def generate_tickets(n: int = 12000, start: str = "2025-01-01", end: str = "2026-06-30",
                     label_noise: float = 0.06, seed: int = 42, score_noise: float = 0.45):
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start).value, pd.Timestamp(end).value
    created = np.sort(rng.uniform(t0, t1, n)).astype("int64")
    created = pd.to_datetime(created)

    weights = np.array([1.0 / (1.0 + 0.35 * max(0.0, c[3] - 3.0)) for c in CATALOG])
    weights /= weights.sum()
    type_idx = rng.choice(len(CATALOG), size=n, p=weights)

    envs = rng.choice(["prod", "staging", "dev"], size=n, p=[0.55, 0.15, 0.30])
    channels = rng.choice(["monitoring", "portal", "email", "phone"], size=n, p=[0.50, 0.28, 0.12, 0.10])
    env_mod = {"prod": 1.2, "staging": -1.0, "dev": -2.8}
    tier_mod = {1: 0.8, 2: 0.0, 3: -1.2}

    rows, scores = [], np.zeros(n)
    svc_df = pd.DataFrame(SERVICES, columns=["service", "tier", "customer_facing", "introduced"])
    svc_df["introduced"] = pd.to_datetime(svc_df["introduced"])

    for i in range(n):
        cat, sub, nature, base, templates = CATALOG[type_idx[i]]
        avail = svc_df[svc_df["introduced"] <= created[i]]
        svc = avail.iloc[rng.integers(len(avail))]
        env, chan = envs[i], channels[i]

        adj = base + env_mod[env] + tier_mod[int(svc["tier"])] + (0.4 if svc["customer_facing"] else 0.0)
        users_true = max(1, int(round(10 ** rng.normal(0.30 * adj - 0.2, 0.6))))
        score = adj + 0.5 * np.log10(users_true + 1) - 0.8 + rng.normal(0, score_noise)
        scores[i] = score

        impact_level = score + rng.normal(0, 1.4)
        level = "high" if impact_level >= 9.0 else "medium" if impact_level >= 6.0 else "low"
        if rng.random() < 0.30:
            level = "none"

        short = templates[rng.integers(len(templates))].format(svc=svc["service"])
        if chan == "monitoring":
            short = f"[{ALERT_SRC[rng.integers(len(ALERT_SRC))]}] {short}"
        elif chan in ("email", "phone"):
            short = f"{USER_PREFIX[rng.integers(len(USER_PREFIX))]} {short}".strip()
        if rng.random() < 0.04:
            short = "URGENT: " + short            # noisy urgency word, unrelated to true severity

        desc_parts = [DETAILS[nature][rng.integers(3)],
                      ENV_PHRASES[env][rng.integers(len(ENV_PHRASES[env]))],
                      IMPACT[level][rng.integers(len(IMPACT[level]))]]
        description = " ".join(p for p in desc_parts if p)

        reported = np.nan if rng.random() < 0.40 else float(max(1, round(users_true * rng.lognormal(0, 0.3))))
        sub_shown = "" if rng.random() < 0.30 else sub          # subcategory often left blank by reporters

        rows.append({
            "ticket_id": f"INC{100000 + i}", "created_at": created[i], "channel": chan,
            "category": cat, "subcategory": sub_shown, "affected_service": svc["service"],
            "environment": env, "short_description": short, "description": description,
            "users_affected": reported,
        })

    df = pd.DataFrame(rows)
    sev = _severity_from_score(scores)

    # simulate inconsistent human triage: shift a share of labels by one level (mostly down)
    flip = rng.random(n) < label_noise
    direction = np.where(rng.random(n) < 0.7, 1, -1)
    sev = np.where(flip, np.clip(sev + direction, 0, 3), sev)
    df["severity"] = np.array(["P1", "P2", "P3", "P4"])[sev]

    # ---- post-triage columns (LEAKAGE): exist in real ITSM exports, unavailable at creation ----
    res = np.exp(rng.normal(np.array([5.8, 5.0, 4.4, 3.8])[sev], 0.6)).round()
    df["resolution_minutes"] = res
    df["closed_at"] = df["created_at"] + pd.to_timedelta(res, unit="m")
    df["assignment_group"] = np.where((sev == 0) & (rng.random(n) < 0.8), "Major-Incident-Team",
                                      np.array(["Platform-OnCall", "App-Support", "IT-Helpdesk", "SecOps"])[rng.integers(0, 4, n)])
    df["close_notes"] = "closed after resolution"

    cmdb = pd.DataFrame({
        "affected_service": svc_df["service"],
        "service_tier": svc_df["tier"],
        "customer_facing": svc_df["customer_facing"].astype(int),
        "owner_team": svc_df["tier"].map(OWNER),
    })
    return df, cmdb
