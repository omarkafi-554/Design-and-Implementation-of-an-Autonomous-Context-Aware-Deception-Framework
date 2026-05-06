import argparse
import csv
import os
import xml.etree.ElementTree as ET

HIGH_RISK   = {"ssh", "telnet", "ftp", "rsh", "rlogin"}
MEDIUM_RISK = {"http", "https", "smtp", "pop3", "imap", "rdp", "vnc"}

def classify_risk(service: str, port: str) -> str:
    s = service.lower()
    if s in HIGH_RISK:
        return "HIGH"
    if s in MEDIUM_RISK:
        return "MEDIUM"
    if port == "2222":
        return "HONEYPOT"
    return "LOW"

def parse_nmap_xml(xml_path: str) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []
    for host in root.findall("host"):
        for port_elem in host.findall(".//port"):
            state = port_elem.find("state")
            if state is None or state.get("state") != "open":
                continue
            port_id = port_elem.get("portid", "")
            proto   = port_elem.get("protocol", "tcp")
            svc     = port_elem.find("service")
            name    = svc.get("name", "unknown")  if svc is not None else "unknown"
            version = svc.get("version", "")      if svc is not None else ""
            product = svc.get("product", "")      if svc is not None else ""
            risk    = classify_risk(name, port_id)
            results.append({
                "port"   : f"{port_id}/{proto}",
                "service": name,
                "version": f"{product} {version}".strip(),
                "risk"   : risk,
            })
    return results

def main():
    parser = argparse.ArgumentParser(description="ASA — Nmap XML parser")
    parser.add_argument("--input",  required=True,  help="Path to Nmap XML output file")
    parser.add_argument("--output", default="docs/asa_report.csv",
                        help="Path for CSV report (default: docs/asa_report.csv)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = parse_nmap_xml(args.input)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["port", "service", "version", "risk"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAttack Surface Report — {len(results)} open ports\n")
    print(f"{'Port':<12} {'Service':<12} {'Version':<30} {'Risk'}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["risk"]):
        print(f"{r['port']:<12} {r['service']:<12} {r['version']:<30} {r['risk']}")

    high_risk = [r for r in results if r["risk"] == "HIGH"]
    if high_risk:
        print(f"\n⚠  HIGH risk services found: {[r['port'] for r in high_risk]}")
        print("   → Recommended action: Deploy Cowrie SSH honeypot")
        print(f"   → iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222")

    print(f"\nReport saved to {args.output}")

if __name__ == "__main__":
    main()
