"""Repeatable, synthetic retail scenario, confined to the demo database."""
from datetime import datetime, timedelta, timezone
from intelligence import product_key

PRODUCTS = [
    ("Wireless Earbuds", "AUD-101", ["bluetooth earbuds", "wireless earphones", "earbuds"]),
    ("USB-C Fast Charger", "PWR-202", ["usb c charger", "type c charger", "fast charger"]),
    ("Everyday Backpack", "BAG-303", ["laptop backpack", "everyday backpack", "work backpack"]),
    ("AirLite Running Shoes — Size 42", "SHOE-42", ["airlite trainers size 42", "airlite running shoes size 42"]),
    ("Insulated Water Bottle", "HOME-505", ["insulated bottle", "thermal water bottle"]),
]
# product index, source, request, customer alias, quantity, budget, reason, days ago
REQUESTS = [
    (0,"whatsapp","Do you have 2 bluetooth earbuds? Budget KES 6,000. I need them today.","Shopper A",2,6000,"out_of_stock",0),
    (0,"website_search","wireless earbuds","Visitor 204",1,None,"not_stocked",1),
    (0,"chatbot","Need wireless earphones but you said they are sold out.","Shopper B",1,3200,"out_of_stock",1),
    (0,"salesperson","Customer asked for 3 wireless earbuds; shelf empty. Budget KES 9,000.","Shopper C",3,9000,"out_of_stock",2),
    (0,"out_of_stock","Out of stock: 2 units Wireless Earbuds. Customer needs them tomorrow.","Shopper D",2,6000,"out_of_stock",3),
    (0,"abandoned_enquiry","Buyer left the earbuds enquiry without a response after hearing the restock date.","Shopper B",1,3000,"abandoned",4),
    (0,"whatsapp","Looking for wireless earbuds. Are they back in stock yet?","Shopper E",1,3500,"out_of_stock",9),
    (1,"rejected_quote","The quote for 4 USB-C fast chargers was too expensive. Budget KES 4,000.","Shopper F",4,4000,"price_too_high",0),
    (1,"whatsapp","Need a type c charger but the price is over budget. I can spend KES 900.","Shopper G",1,900,"price_too_high",1),
    (1,"chatbot","The fast charger costs more than I can afford.","Shopper H",1,1000,"price_too_high",2),
    (1,"rejected_quote","Customer declined the USB-C charger quote and chose another supplier.","Shopper I",2,2400,"quote_rejected",3),
    (1,"salesperson_voice","A customer asked for two fast chargers but wanted a cheaper option.","Shopper J",2,2000,"price_too_high",5),
    (1,"website_search","usb c charger",None,1,None,"not_stocked",11),
    (2,"whatsapp","Need an everyday backpack by tomorrow but delivery takes a week.","Shopper K",1,4200,"lead_time",0),
    (2,"salesperson_voice","Customer needs a laptop backpack for travel today; delivery misses the deadline.","Shopper L",1,4500,"lead_time",1),
    (2,"chatbot","Can I collect a work backpack? The delivery time is too long.","Shopper M",1,4000,"lead_time",2),
    (2,"abandoned_enquiry","Shopper never replied after receiving the backpack details.","Shopper N",1,3800,"abandoned",9),
    (3,"website_search","airlite trainers size 42","Visitor 910",1,None,"not_stocked",0),
    (3,"unavailable_product","Looking for AirLite running shoes size 42. You do not carry this size.","Shopper O",1,6500,"not_stocked",2),
    (3,"whatsapp","Do you have AirLite trainers size 42? Budget KES 7,000.","Shopper P",1,7000,"not_stocked",4),
    (4,"abandoned_enquiry","Customer ghosted us after asking for an insulated bottle.","Shopper Q",1,1800,"abandoned",1),
    (4,"chatbot","Need 2 insulated water bottles, but checkout was never completed.","Shopper R",2,3600,"abandoned",2),
    (4,"whatsapp","Looking for an insulated bottle; budget USD 18. No reply to our follow-up.","Shopper S",1,18,"abandoned",3),
    (4,"salesperson","Customer wanted an insulated bottle but it was sold out.","Shopper T",1,2000,"out_of_stock",8),
]


def seed(connection, save_signal):
    now = datetime.now(timezone.utc)
    for name, sku, aliases in PRODUCTS:
        existing = connection.execute("SELECT id FROM catalog WHERE sku=?", (sku,)).fetchone()
        if existing:
            continue
        cursor = connection.execute("INSERT INTO catalog(name,sku,created_at) VALUES(?,?,?)", (name, sku, now.isoformat()))
        for alias in {product_key(a) for a in [name, *aliases]}:
            connection.execute("INSERT OR IGNORE INTO product_aliases VALUES(?,?)", (alias, cursor.lastrowid))
    connection.commit()
    added = 0
    for index, (product, source, text, customer, qty, budget, reason, days) in enumerate(REQUESTS):
        signal = save_signal(text, source, customer, external_id=f"demo-v1-{index}", fields=dict(
            product=PRODUCTS[product][0], qty=qty, budget=budget, reason=reason))
        if signal["deduplicated"]:
            continue
        added += 1
        connection.execute("UPDATE signals SET ts=?,reason_engine='demo' WHERE id=?",
                           ((now-timedelta(days=days, hours=index%4)).isoformat(), signal["id"]))
        if index == 23:
            connection.execute("UPDATE signals SET status='recovered',owner='Alex',next_action='Customer collected after restock',recovered_amount=1900,recovered_currency='KES' WHERE id=?", (signal["id"],))
        elif index == 15:
            connection.execute("UPDATE signals SET status='follow_up',owner='Sam',next_action='Check same-day collection availability' WHERE id=?", (signal["id"],))
    connection.commit()
    return dict(added=added, message=f"{added} synthetic requests added to the demo workspace. Live records are separate.")
