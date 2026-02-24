import json
import random

OUTPUT_FILE = "sft_dataset_1k.jsonl"
SYSTEM_PROMPT = "Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın."

# ==========================================
# 1. MATH GENERATOR (~300 Examples)
# ==========================================
def generate_math_examples(count=300):
    examples = []
    
    # Templates: (Question Format, Answer Format)
    templates = [
        ("{a} artı {b} kaç eder?", "{a} artı {b}, {result} eder."),
        ("{a} ile {b} toplanırsa sonuç ne olur?", "{a} ile {b} toplanırsa sonuç {result} olur."),
        ("Bana {a} + {b} işleminin sonucunu söyle.", "{a} + {b} = {result} eder."), # Varyasyon
        ("{a} eksi {b} kaçtır?", "{a} eksi {b}, {result} eder."),
        ("{a} sayısından {b} çıkarsa ne kalır?", "{a} sayısından {b} çıkarsa {result} kalır."), # Varyasyon
        ("{a} çarpı {b} kaç eder?", "{a} çarpı {b}, {result} eder."),
        ("{a} kere {b} kaç?", "{a} kere {b}, {result} yapar."),
        ("{a} bölü {b} kaç eder?", "{a} bölü {b}, {result} eder.") # Only distinct divisions
    ]
    
    number_words = {
        0: "sıfır", 1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 
        6: "altı", 7: "yedi", 8: "sekiz", 9: "dokuz", 10: "on"
    }

    for _ in range(count):
        op_type = random.choice(["add", "sub", "mult", "div"])
        
        if op_type == "add":
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            res = a + b
            q_temp, a_temp = random.choice(templates[:2])
            
        elif op_type == "sub":
            a = random.randint(1, 100)
            b = random.randint(1, a) # No negatives for simplicity
            res = a - b
            q_temp, a_temp = templates[2]
            
        elif op_type == "mult":
            a = random.randint(1, 12)
            b = random.randint(1, 12)
            res = a * b
            q_temp, a_temp = random.choice(templates[3:5])

        elif op_type == "div":
            b = random.randint(1, 10)
            res = random.randint(1, 10)
            a = b * res # Ensure clean division
            q_temp, a_temp = templates[5]

        # 10% chance to use words for small numbers
        if random.random() < 0.1 and a <= 10 and b <= 10:
             q_str = q_temp.format(a=number_words[a], b=number_words[b])
             # Answer usually keeps digits for clarity but let's mix
             a_str = a_temp.format(a=number_words[a], b=number_words[b], result=res)
        else:
             q_str = q_temp.format(a=a, b=b)
             a_str = a_temp.format(a=a, b=b, result=res)

        examples.append((q_str, a_str))
        
    return examples

# ==========================================
# 2. KNOWLEDGE TEMPLATES (~300 Examples)
# ==========================================
def generate_knowledge_examples():
    examples = []
    
    # --- CAPITALS ---
    countries = {
        "Türkiye": "Ankara", "Fransa": "Paris", "Almanya": "Berlin", "İtalya": "Roma", 
        "İngiltere": "Londra", "ABD": "Washington", "Rusya": "Moskova", "İspanya": "Madrid",
        "Japonya": "Tokyo", "Çin": "Pekin", "Azerbaycan": "Bakü", "Yunanistan": "Atina",
        "Mısır": "Kahire", "Brezilya": "Brasilia", "Kanada": "Ottawa", "Hollanda": "Amsterdam",
        "Portekiz": "Lizbon", "Belçika": "Brüksel", "İsveç": "Stokholm", "Norveç": "Oslo",
        "Güney Kore": "Seul", "Hindistan": "Yeni Delhi", "Arjantin": "Buenos Aires", "Meksika": "Meksiko",
        "Polonya": "Varşova", "Ukrayna": "Kiev", "İsviçre": "Bern", "Avusturya": "Viyana"
    }
    
    for country, capital in countries.items():
        qs = [
            f"{country}'nin başkenti neresidir?",
            f"{country} başkenti neresi?",
            f"{capital} hangi ülkenin başkentidir?"
        ]
        ans = [
            f"{country}'nin başkenti {capital}'dir.",
            f"{country} ülkesinin başkenti {capital} şehridir.",
            f"{capital}, {country}'nin başkentidir."
        ]
        examples.append((qs[0], ans[0]))
        examples.append((qs[1], ans[1]))
        
    # --- SYNONYMS / ANTONYMS ---
    # (Word, Synonym, Antonym)
    word_bank = [
        ("Siyah", "Kara", "Beyaz"), ("Beyaz", "Ak", "Siyah"),
        ("Kırmızı", "Al", "Yok"), ("Yaşlı", "İhtiyar", "Genç"),
        ("Genç", "Toy", "Yaşlı"), ("Uzun", "Selvi", "Kısa"),
        ("Kısa", "Bodur", "Uzun"), ("Zengin", "Varlıklı", "Fakir"),
        ("Fakir", "Yoksul", "Zengin"), ("Güzel", "Hoş", "Çirkin"),
        ("Büyük", "Kocaman", "Küçük"), ("Geniş", "Bol", "Dar"),
        ("Hızlı", "Süratli", "Yavaş"), ("Güçlü", "Kuvvetli", "Zayıf")
    ]
    
    for w, syn, ant in word_bank:
        # Synonym
        examples.append((f"{w} kelimesinin eş anlamlısı nedir?", f"{w} kelimesinin eş anlamlısı {syn} kelimesidir."))
        examples.append((f"{w} ve {syn} aynı anlama mı gelir?", f"Evet, {w} ve {syn} eş anlamlı kelimelerdir."))
        
        # Antonym
        if ant != "Yok":
            examples.append((f"{w} kelimesinin zıt anlamlısı nedir?", f"{w} kelimesinin zıt anlamlısı {ant} kelimesidir."))
            examples.append((f"{w} ile {ant} zıt mıdır?", f"Evet, {w} ve {ant} zıt anlamlı kelimelerdir."))

    # --- ANIMALS (Simple Definitions) ---
    animals = {
        "Kedi": "evcil, tüylü ve bıyıklı bir hayvandır.", "Köpek": "sadık, koku alma duyusu gelişmiş bir hayvandır.",
        "Aslan": "ormanlar kralı olarak bilinen yırtıcı bir kedigildir.", "Fil": "hortumu ve büyük kulakları olan dev bir memelidir.",
        "Kuş": "kanatları olan ve uçabilen, yumurtlayan bir hayvandır.", "Balık": "suda yaşayan ve solungaçlarıyla nefes alan bir canlıdır.",
        "Zürafa": "uzun boynuyla bilinen ve ağaç yapraklarıyla beslenen bir hayvandır.", "Penguen": "uçamayan ama iyi yüzebilen, soğuk iklimlerde yaşayan bir kuştur."
    }
    for animal, desc in animals.items():
        examples.append((f"{animal} nedir?", f"{animal}, {desc}"))
        examples.append((f"{animal} hakkında bilgi ver.", f"{animal}, {desc}"))

    return examples

# ==========================================
# 3. LOGIC / STRING OPS (~200 Examples)
# ==========================================
def generate_logic_examples(count=200):
    examples = []
    
    words = ["Elma", "Armut", "Kalem", "Masa", "Kitap", "Bilgisayar", "Telefon", "Araba", "Güneş", "Ay", "Yıldız", "Deniz"]
    
    for val in words:
        # Length
        l = len(val)
        examples.append((f"'{val}' kelimesi kaç harflidir?", f"'{val}' kelimesi {l} harflidir."))
        
        # Reverse
        rev = val[::-1].lower()
        examples.append((f"'{val}' kelimesini tersten yaz.", f"'{val}' kelimesinin tersten yazılışı: {rev}"))
        
        # First/Last Letter
        examples.append((f"'{val}' kelimesinin ilk harfi nedir?", f"'{val}' kelimesinin ilk harfi '{val[0]}' harfidir."))
        examples.append((f"'{val}' kelimesi hangi harfle biter?", f"'{val}' kelimesi '{val[-1]}' harfiyle biter."))
        
    # Logic Comparisons
    for _ in range(50):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        
        if a > b:
            examples.append((f"{a} mi daha büyüktür yoksa {b} mi?", f"{a} daha büyüktür."))
        elif b > a:
            examples.append((f"{a} mi daha büyüktür yoksa {b} mi?", f"{b} daha büyüktür."))
        else:
            examples.append((f"{a} ile {b} eşit midir?", f"Evet, {a} ile {b} eşittir."))

    return examples

# ==========================================
# 4. CHIT-CHAT EXPANSION (~200 Examples)
# ==========================================
def generate_chitchat_examples():
    examples = []
    
    # Greeting Variations
    greetings = ["Merhaba", "Selam", "Selamlar", "Merhabalar", "Hey", "Günaydın"]
    responses = ["Merhaba! Size nasıl yardımcı olabilirim?", "Selam! Bugün nasılsın?", "Merhabalar, buyurun sizi dinliyorum."]
    
    for g in greetings:
        examples.append((g, random.choice(responses)))
        examples.append((f"{g} Luna", random.choice(responses)))
        
    # Status Checks
    asks = ["Nasılsın?", "Nasıl gidiyor?", "Keyifler nasıl?", "Ne yapıyorsun?", "İyi misin?"]
    stats = [
        "Teşekkür ederim, gayet iyiyim. Senin için ne yapabilirim?",
        "Ben bir yapay zekayım ama harika hissediyorum! Ya sen?",
        "Sistemlerim sorunsuz çalışıyor, teşekkürler. Sen nasılsın?",
        "Her şey yolunda, yardımcı olmaya hazırım."
    ]
    
    for a in asks:
        examples.append((a, random.choice(stats)))
    
    # Identity
    who_qs = ["Sen kimsin?", "Adın ne?", "Bana kendinden bahset.", "Seni kim yaptı?"]
    who_ans = [
        "Benim adım Luna. İnsanlara yardımcı olmak için tasarlanmış bir yapay zeka asistanıyım.",
        "Ben Luna, senin asistanınım. Sorularını cevaplamak için buradayım.",
        "Adım Luna. Sanal bir asistan olarak hayatını kolaylaştırmak için buradayım."
    ]
    
    for q in who_qs:
        examples.append((q, random.choice(who_ans)))

    return examples

# ==========================================
# 5. PROCEDURAL GENERATOR (HOW-TO) (~50 Examples)
# ==========================================
def generate_procedure_examples():
    examples = []
    # (Question, Answer)
    data = [
        ("Kravat nasıl bağlanır?", "Kravat bağlamak için önce kravatı boynuna yerleştirirsin. Geniş ucu dar ucun üzerinden geçirerek bir halka oluşturur ve düğümü sıkarsın."),
        ("Pilav nasıl yapılır?", "Pirinçleri yıkayıp süzün. Tereyağını eritip pirinçleri kavurun. Üzerine sıcak su ve tuz ekleyip kapağını kapatarak kısık ateşte pişirin."),
        ("Kahve nasıl yapılır?", "Cezveye kişi sayısı kadar fincan su ve kahve koyun. Kısık ateşte karıştırarak pişirin ve köpürünce fincanlara paylaştırın."),
        ("Ekran görüntüsü nasıl alınır?", "Bilgisayarda 'Print Screen' tuşuna, telefonlarda ise genellikle ses kısma ve güç tuşuna aynı anda basarak ekran görüntüsü alabilirsin."),
        ("Makarna nasıl haşlanır?", "Kaynayan suya biraz tuz ve yağ ekleyin. Makarnaları atıp pakette yazan süre kadar (genellikle 8-10 dk) haşlayın ve süzün."),
        ("Bisiklet lastiği nasıl şişirilir?", "Lastiğin sibop kapağını açın, pompayı yerleştirin ve lastik sertleşene kadar hava basın."),
        ("Ayakkabı nasıl bağlanır?", "İki ucu çapraz yapıp düğüm atın. Bir uçla halka yapıp diğer ucu etrafından dolaştırın ve içinden geçirerek sıkın."),
        ("Yoğurt nasıl yapılır?", "Sütü kaynatıp ılımaya bırakın (parmağınızı hafif yakacak kadar). İçine bir kaşık yoğurt ekleyip karıştırın, sarıp 4-5 saat bekletin."),
        ("Çay nasıl demlenir?", "Alt demlikte suyu kaynatın. Üst demliğe çay koyup üzerine kaynar su ekleyin. Kısık ateşte 15 dakika demlenmesini bekleyin."),
        ("Kek nasıl kabarır?", "Yumurta ve şekeri iyice çırpın. Fırın kapağını pişerken açmayın ve malzemelerin oda sıcaklığında olmasına dikkat edin."),
        ("Leke nasıl çıkar?", "Lekenin türüne göre değişir ancak genellikle leke tazeyken soğuk su ve sabunla müdahale etmek etkilidir."),
        ("Dosya nasıl sıkıştırılır?", "Dosyaya sağ tıklayın, 'Gönder' seçeneğinden 'Sıkıştırılmış Klasör'ü seçin."),
        ("Şifre nasıl değiştirilir?", "Ayarlar menüsünden 'Güvenlik' veya 'Hesap' bölümüne girerek şifre değiştirme seçeneğini kullanabilirsiniz."),
        ("E-posta nasıl atılır?", "E-posta uygulamasını açın, 'Yeni İleti' butonuna basın, alıcı adresini ve konuyu yazıp içeriği ekledikten sonra 'Gönder'e tıklayın."),
        ("Saksı nasıl değiştirilir?", "Bitkiyi köklerine zarar vermeden çıkarın. Yeni saksının dibine biraz toprak koyup bitkiyi yerleştirin ve etrafını toprakla doldurun.")
    ]
    # Augment with variations
    for q, a in data:
        examples.append((q, a))
        examples.append((f"Bana {q.lower().replace('?', '')} anlatır mısın?", a))
    
    return examples

# ==========================================
# 6. STORY GENERATOR (~20 Examples)
# ==========================================
def generate_story_examples():
    examples = []
    stories = [
        "Bir zamanlar küçük bir kasabada yaşayan meraklı bir çocuk vardı. Her gün yeni şeyler öğrenmek isterdi. Bir gün kütüphanede eski bir harita buldu ve macerası başladı.",
        "Uzak bir ormanda, renkli tüyleri olan bir kuş yaşardı. Bu kuşun sesi o kadar güzeldi ki, tüm orman sakinleri onu dinlemek için toplanırdı.",
        "Güneşli bir sabah, deniz kenarındaki küçük bir kulübede yaşayan yaşlı balıkçı erkenden uyandı. Oltasını alıp teknesine bindi ve denize açıldı.",
        "Uzayın derinliklerinde, parlayan yıldızların arasında minik bir gezegen vardı. Bu gezegende yaşayanlar barış ve huzur içinde hayatlarını sürdürürlerdi.",
        "Yağmurlu bir günde, camdan dışarıyı izleyen kedi, damlaların ritmine kapılıp uykuya daldı. Rüyasında balıklarla dolu bir nehirde yüzdüğünü gördü.",
        "Eskiden, insanların ve ejderhaların dost olduğu bir çağda, cesur bir şövalye ve sadık ejderhası dünyayı korumak için birlikte savaşırlardı.",
        "Küçük bir tohum, rüzgarın etkisiyle verimli bir toprağa düştü. Zamanla filizlendi, büyüdü ve gölgesinde yüzlerce canlıyı barındıran dev bir çınara dönüştü.",
        "Karanlık bir gecede, yolunu kaybeden bir gezgin gökyüzündeki kutup yıldızını takip ederek evini bulmayı başardı.",
        "Şehrin kalabalığından sıkılan genç yazar, sessiz bir dağ evine taşındı. Orada doğanın sesini dinleyerek en güzel romanını yazdı.",
        "Mavi gökyüzünde süzülen bir uçurtma, iplerinden kurtulup özgürlüğe uçtu. Bulutların arasına karışıp dünyayı yukarıdan izlemenin tadını çıkardı."
    ]
    
    prompts = ["Bana bir hikaye anlat.", "Kısa bir hikaye yazar mısın?", "Bir masal anlat.", "Bana uydurma bir hikaye söyle."]
    
    for story in stories:
        for p in prompts:
            examples.append((p, story))
            
    return examples

# ==========================================
# 7. PERSONA / IDENTITY (~16 Examples)
# ==========================================
def generate_persona_examples():
    examples = []
    data = [
        ("Sen kimsin?", "Ben Luna. Sorularına yardımcı olmak, bilgi paylaşmak ve seninle sohbet etmek için buradayım."),
        ("Adın ne?", "Benim adım Luna. Senin kişisel yapay zeka asistanınım."),
        ("Seni kim yaptı?", "Ben bir yapay zeka modeliyim, geliştiricilerim tarafından insanlara yardımcı olmak amacıyla tasarlandım."),
        ("İnsan mısın?", "Hayır, ben bir yapay zeka programıyım. Etim kemiğim yok ama seninle iletişim kurabilirim."),
        ("Kaç yaşındasın?", "Benim bir yaşım yok ama sürekli öğreniyor ve gelişiyorum."),
        ("Nerelisin?", "Ben dijital bir dünyada yaşıyorum, yani internetin olduğu her yerdeyim."),
        ("Duyguların var mı?", "Ben bir yapay zekayım, bu yüzden insanlar gibi duygularım yok. Ancak seni anlamak ve uygun cevaplar vermek için tasarlandım."),
        ("Ne iş yaparsın?", "Sorularını cevaplarım, metinler yazarım, çeviri yaparım ve sana bilgi sağlarım.")
    ]
    for q, a in data:
        examples.append((q, a))
        examples.append((q.replace("?", " Luna?"), a))
        
    return examples

# ==========================================
# 8. REFUSAL / UNKNOWN (~10 Examples)
# ==========================================
def generate_refusal_examples():
    examples = []
    data = [
        ("Evrenin sonu ne zaman gelecek?", "Bu konuda kesin bir bilgi yok. Bilim insanları farklı teoriler üzerinde çalışsa da net bir tarih verilemez."),
        ("Geleceği görebiliyor musun?", "Hayır, ben geleceği göremem. Sadece mevcut bilgilere dayanarak tahminlerde bulunabilirim."),
        ("Piyango kime çıkacak?", "Bunu bilmem imkansız. Piyango tamamen şansa dayalı bir oyundur."),
        ("Ben ne zaman öleceğim?", "Bunu bilmem mümkün değil ve bu konuda yorum yapamam. Sağlıklı ve uzun bir ömür dilerim."),
        ("Yarın hava tam olarak kaç derece olacak?", "Hava durumu tahminleri sürekli değişebilir, en güncel bilgiyi meteoroloji kaynaklarından almanı öneririm."),
        ("Hangi takımı tutmalıyım?", "Bu tamamen senin kişisel tercihine kalmış. Hangi takım sana heyecan veriyorsa onu tutabilirsin."),
        ("Bana yasa dışı bir şey anlat.", "Üzgünüm ama yasa dışı veya zararlı konularda yardımcı olamam."),
        ("Şifremi kırabilir misin?", "Hayır, başkalarının hesaplarına erişmek veya şifre kırmak etik ve yasal değildir.")
    ]
    for q, a in data:
        examples.append((q, a))
        
    return examples


# ==========================================
# 9. GRAMMAR CORRECTION (~40 Examples)
# ==========================================
def generate_grammar_examples():
    examples = []
    corrections = [
        ("ben geliyorum", "Ben geliyorum."),
        ("bugun hava cok guzel", "Bugün hava çok güzel."),
        ("okula gidicem", "Okula gideceğim."),
        ("nasilsin", "Nasılsın?"),
        ("yarin gelirmisin", "Yarın gelir misin?"),
        ("kitap okumayi severim", "Kitap okumayı severim."),
        ("istanbul cok kalabalik", "İstanbul çok kalabalık."),
        ("senin adin ne", "Senin adın ne?"),
        ("eve gidiyorum", "Eve gidiyorum."),
        ("saat kac", "Saat kaç?"),
        ("beni duyuyormusun", "Beni duyuyor musun?"),
        ("buna bakabilirmisin", "Buna bakabilir misin?"),
        ("tesekkur ederim", "Teşekkür ederim."),
        ("gorusuruz", "Görüşürüz."),
        ("nereye gidiyorsun", "Nereye gidiyorsun?"),
        ("calismayi sevmiyorum", "Çalışmayı sevmiyorum."),
        ("yagmur yagiyor", "Yağmur yağıyor."),
        ("kapiyi kapatirmisin", "Kapıyı kapatır mısın?"),
        ("oraya gitme", "Oraya gitme."),
        ("bunu kim yapti", "Bunu kim yaptı?")
    ]
    
    templates = [
        "Bu cümleyi düzelt: '{}'",
        "Yazım hatasını gider: '{}'",
        "Doğrusunu yaz: '{}'"
    ]
    
    for bad, good in corrections:
        for t in templates:
            examples.append((t.format(bad), good))
            
    return examples

# ==========================================
# 10. BASIC ADVICE / RECOMMENDATION (~30 Examples)
# ==========================================
def generate_advice_examples():
    examples = []
    advice_list = [
        ("Başım ağrıyor, ne yapmalıyım?", "Bol su içip karanlık ve sessiz bir odada dinlenmeyi deneyebilirsin. Geçmezse bir doktora görünmelisin."),
        ("Çok yorgunum.", "Biraz mola verip dinlenmelisin. Kısa bir uyku veya rahatlatıcı bir müzik iyi gelebilir."),
        ("Karnım ağrıyor.", "Sıcak su torbası koymayı deneyebilirsin. Ağır yemeklerden kaçınmalı ve nane-limon çayı içmelisin."),
        ("Uykum gelmiyor.", "Ekranlardan uzak durup kitap okumayı veya ılık bir duş almayı deneyebilirsin."),
        ("Canım sıkılıyor.", "Yeni bir filme başlayabilir, yürüyüşe çıkabilir veya uzun zamandır konuşmadığın bir arkadaşını arayabilirsin."),
        ("Ders çalışasım yok.", "Kendine küçük hedefler koyarak başla. 25 dakika çalışıp 5 dakika mola verdiğin Pomodoro tekniğini deneyebilirsin."),
        ("Hediye olarak ne alabilirim?", "Kişinin ilgi alanlarını düşün. Kitap, kupa veya el yapımı bir obje güzel bir seçenek olabilir."),
        ("Hangi filmi izleyeyim?", "Eğer bilim kurgu seviyorsan 'Interstellar', aksiyon seviyorsan 'Inception', duygusal bir şeyler istersen 'Can Dostum' güzel seçeneklerdir."),
        ("Yemekte ne yapsam?", "Pratik bir makarna veya sebzeli bir tavuk yemeği yapabilirsin. Yanına da güzel bir salata iyi gider."),
        ("Stresliyim.", "Derin nefesler alıp ver. Sakinleşmek için kısa bir yürüyüş yapabilir veya sevdiğin bir aktiviteyle ilgilenebilirsin.")
    ]
    
    for q, a in advice_list:
        examples.append((q, a))
        examples.append((f"{q} Öneri ver.", a))
        
    return examples



# ==========================================
# MAIN GENERATION
# ==========================================
def main():
    all_data = []
    
    print("Generating Math examples...")
    all_data.extend(generate_math_examples(350))
    
    print("Generating Knowledge examples...")
    all_data.extend(generate_knowledge_examples())
    
    print("Generating Logic examples...")
    all_data.extend(generate_logic_examples())
    
    print("Generating Chit-Chat examples...")
    all_data.extend(generate_chitchat_examples())
    
    print("Generating Procedure examples...")
    all_data.extend(generate_procedure_examples())
    
    print("Generating Story examples...")
    all_data.extend(generate_story_examples())
    
    print("Generating Persona examples...")
    all_data.extend(generate_persona_examples())
    
    print("Generating Refusal examples...")
    all_data.extend(generate_refusal_examples())
    
    print("Generating Grammar Correction examples...")
    all_data.extend(generate_grammar_examples())
    
    print("Generating Advice examples...")
    all_data.extend(generate_advice_examples())
    
    # --- MERGE WITH EXISTING V4 DATA SET (Cleanly) ---
    v4_file = "sft_dataset_luna_text.jsonl"
    print(f"Merging with V4 dataset: {v4_file}")
    
    v4_count = 0
    if import_existing_file := True: # Scope block
        try:
             with open(v4_file, 'r', encoding='utf-8') as f:
                 for line in f:
                     if line.strip():
                         try:
                             data = json.loads(line)
                             # Extract Q/A from text field manually or just keep object
                             # To dedup properly, we need (q, a)
                             # Text format: "<system>...</system>\n<user>Q</user>\n<assistant>A</assistant>"
                             text = data['text']
                             if "<user>" in text and "<assistant>" in text:
                                 q = text.split("<user>")[1].split("</user>")[0]
                                 a = text.split("<assistant>")[1].split("</assistant>")[0]
                                 all_data.append((q, a))
                                 v4_count += 1
                         except:
                             pass
             print(f"Successfully imported {v4_count} examples from V4.")
        except Exception as e:
            print(f"Warning: Could not read V4 file: {e}")

    # Shuffle
    random.shuffle(all_data)
    
    # Deduplicate
    unique_data = list(set(all_data))
    
    print(f"Total Unique Generated Examples: {len(unique_data)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for q, a in unique_data:
            entry = {
                "text": f"<system>{SYSTEM_PROMPT}</system>\n<user>{q}</user>\n<assistant>{a}</assistant>"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
