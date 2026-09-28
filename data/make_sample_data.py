import csv
import random
from pathlib import Path

rng = random.Random(7)
HERE = Path(__file__).parent

NAMES = ["Thabo", "Lerato", "Sipho", "Ayanda", "Pieter", "Naledi", "Kagiso", "Zanele", "Johan", "Palesa"]
BANKS = [("Standard Bank", "standardbank.co.za"), ("FNB", "fnb.co.za"), ("Capitec", "capitecbank.co.za"),
         ("Absa", "absa.co.za"), ("Nedbank", "nedbank.co.za")]
SHOPS = [("Takealot", "takealot.com"), ("Woolworths", "woolworths.co.za"), ("Checkers", "checkers.co.za")]
SERVICES = [("Netflix", "netflix.com"), ("Showmax", "showmax.com"), ("Spotify", "spotify.com"),
            ("Microsoft 365", "microsoft.com"), ("Vodacom", "vodacom.co.za"), ("MTN", "mtn.co.za")]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
BAD_TLDS = ["xyz", "top", "tk", "click", "ru", "link", "live"]
NEW_TLDS = ["app", "dev", "online", "site", "io"]
FILLER = ["Kind regards.", "Thank you.", "Have a good day.", "Best wishes.", "Regards, the team.", ""]


def num(): return str(rng.randint(10000, 99999))
def hexs(n=10): return "".join(rng.choice("0123456789abcdef") for _ in range(n))
def ip(): return ".".join(str(rng.randint(11, 250)) for _ in range(4))
def pick(x): return rng.choice(x)


def typo(domain):
    name, rest = domain.split(".", 1)
    swaps = {"a": "4", "o": "0", "l": "1", "i": "1", "e": "3"}
    for ch, rep in swaps.items():
        if ch in name:
            return name.replace(ch, rep, 1) + "." + rest
    return name + "-za." + rest


def bad_url(domain):
    brand = domain.split(".")[0]
    return pick([
        f"http://{brand}-secure-login.{pick(BAD_TLDS)}/verify?id={hexs()}",
        f"http://{ip()}/{brand}/login.php",
        f"https://bit.ly/{hexs(6)}",
        f"http://www.{typo(domain)}/account/update",
        f"http://{brand}.com.account-verify.{pick(BAD_TLDS)}/signin",
        f"http://secure-{brand}@{ip()}/confirm",
        f"https://{brand}-support-{num()}.com/billing/update",
    ])


def legit_email():
    n, (bank, bdom), (shop, sdom), (svc, vdom) = pick(NAMES), pick(BANKS), pick(SHOPS), pick(SERVICES)
    return pick([
        f"Hi {n}, your order #{num()} from {shop} has shipped. Track it here: https://www.{sdom}/orders/{num()}",
        f"Hello {n}, the team meeting has moved to {pick(DAYS)} at 10:00. Agenda: https://docs.spu.ac.za/agenda/{num()}",
        f"Dear {n}, your {bank} statement for this month is available. Log in to the {bank} app to view it.",
        f"Urgent: server maintenance starts tonight at 22:00. Please save your work before then. IT Support",
        f"Hi {n}, you requested a password reset. If this was you use https://login.{vdom}/reset?token={hexs()} otherwise ignore this email.",
        f"Your {svc} subscription renews on {pick(DAYS)}. Manage your plan at https://www.{vdom}/account",
        f"Hey {n}, lunch on {pick(DAYS)}? Let me know if 13:00 works.",
        f"Reminder: assignment {rng.randint(1, 5)} is due on {pick(DAYS)}. Submit on https://learn.spu.ac.za/course/{num()}",
        f"This week at {shop}: new arrivals and 20% off selected items. Browse https://www.{sdom}/sale Unsubscribe https://www.{sdom}/unsubscribe",
        f"Congratulations {n}! You have been selected for the tutor position. HR will contact you about the contract.",
        f"Hi {n}, attached are the minutes from yesterday's meeting. Let me know if I missed anything.",
        f"Your {bank} card ending in {rng.randint(1000, 9999)} was used for R{rng.randint(50, 900)} at {shop}. If you do not recognise this, call the number on the back of your card.",
        f"{svc}: we noticed a new sign in to your account from Windows. If this was you, no action is needed. https://www.{vdom}/security",
    ]) + " " + pick(FILLER)


def phishing_email():
    n, (bank, bdom), (svc, vdom) = pick(NAMES), pick(BANKS), pick(SERVICES)
    return pick([
        f"Dear customer, your {bank} account has been suspended due to unusual activity. Verify your account immediately at {bad_url(bdom)} or it will be closed within 24 hours.",
        f"URGENT: your mailbox is full. Click {bad_url(vdom)} to upgrade your storage now or lose all your emails.",
        f"Congratulations! You have won a R{rng.randint(5, 50)}000 {pick(['voucher', 'prize', 'gift card'])}. Claim your reward now: {bad_url(vdom)}",
        f"{svc}: payment failed. Update your billing details here {bad_url(vdom)} to avoid suspension of your account.",
        f"Hello, I am a lawyer representing a late client. You are entitled to R{rng.randint(1, 9)} million. Reply with your bank details and ID number to proceed.",
        f"Security alert: someone tried to sign in to your {svc} account. If this was not you confirm your password at {bad_url(vdom)}",
        f"Hi {n}, please process the invoice payment today, I am in a meeting and cannot talk. Reply once done, it is urgent.",
        f"Your SARS tax refund of R{rng.randint(1000, 9000)} is ready. Submit your banking details at {bad_url('sars.gov.za')} to receive it.",
        f"Dear {n}, your {bank} OTP has been blocked. Log in at {bad_url(bdom)} and enter your PIN to unlock it.",
        f"Hi {n}, here is the document you requested: {bad_url(vdom)}",
    ]) + " " + pick(FILLER)


def zero_day_phishing():
    n = pick(NAMES)
    word = pick(["walletsync", "docuview", "parcel-clear", "hr-portal", "chainlink-verify", "fileshare-now"])
    url = f"https://{word}-{hexs(4)}.{pick(NEW_TLDS)}/{pick(['connect', 'view', 'pay', 'open'])}/{hexs(8)}"
    return pick([
        f"Hi {n}, your crypto wallet needs re-synchronisation after the network upgrade. Connect your wallet at {url} to keep your tokens.",
        f"Customs notice: parcel {num()} is held at the depot. A clearance fee of R{rng.randint(20, 90)} is outstanding. Settle it via {url}",
        f"{n}, your manager shared 'Salary Adjustments.xlsx' with you. Open in the secure viewer: {url}",
        f"Your AI assistant trial is ending. Keep premium access by re-entering card information: {url}",
        f"Seed phrase check: the exchange is migrating accounts. Paste your 12-word recovery phrase at {url}",
    ])


def zero_day_legit():
    n = pick(NAMES)
    return pick([
        f"Hi {n}, your crypto exchange monthly report is ready in the app. No action is needed.",
        f"Parcel {num()} was delivered to your locker. Collect it with code {rng.randint(1000, 9999)}.",
        f"{n}, your manager shared 'Team schedule.docx' with you: https://spu.sharepoint.com/sites/team/{hexs(8)}",
        f"Your AI assistant plan renews next month. Manage it at https://www.microsoft.com/account",
        f"The exchange will be offline for upgrades on {pick(DAYS)} night. Your tokens are safe; nothing to do.",
    ])


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)


def build(n, make_legit, make_phish):
    rows = [(make_legit(), 0) for _ in range(n // 2)] + [(make_phish(), 1) for _ in range(n // 2)]
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    train = build(3000, legit_email, phishing_email)
    stream = build(200, zero_day_legit, zero_day_phishing)
    seen_train = {t for t, _ in train}
    seen_stream = {t for t, _ in stream}
    test = [r for r in build(1200, legit_email, phishing_email) if r[0] not in seen_train][:800]
    zero_test = [r for r in build(600, zero_day_legit, zero_day_phishing) if r[0] not in seen_stream][:300]
    write(HERE / "train.csv", train)
    write(HERE / "test.csv", test)
    write(HERE / "zero_day_stream.csv", stream)
    write(HERE / "zero_day_test.csv", zero_test)

    samples = HERE / "samples"
    samples.mkdir(exist_ok=True)
    (samples / "phishing_example.txt").write_text(
        "Subject: Account suspended\n\nDear customer,\n\nYour Capitec account has been suspended due to "
        "unusual activity. Verify your account immediately at "
        "<a href=\"http://capitec-secure-login.xyz/verify?id=88af\">www.capitecbank.co.za</a> "
        "or it will be closed within 24 hours.\n\nCapitec Security Team\n", encoding="utf-8")
    (samples / "safe_example.txt").write_text(
        "Subject: Your order has shipped\n\nHi Lerato,\n\nYour order #48213 from Takealot has shipped. "
        "Track it here: https://www.takealot.com/orders/48213\n\nThanks for shopping with us.\n", encoding="utf-8")
    (samples / "zero_day_example.txt").write_text(
        "Subject: Action needed\n\nHi Sipho, your crypto wallet needs re-synchronisation after the network "
        "upgrade. Connect your wallet at https://walletsync-9f2c.app/connect/1a2b3c4d to keep your tokens.\n",
        encoding="utf-8")
    print("Created train.csv, test.csv, zero_day_stream.csv, zero_day_test.csv and samples/")
