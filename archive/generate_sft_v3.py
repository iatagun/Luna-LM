import json
import random

OUTPUT_FILE = r"c:\Users\user\OneDrive\Belgeler\GitHub\Luna-LM\sft_dataset_luna_text.jsonl"

SYSTEM_PROMPT = "Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın."

# --- 1. PREVIOUS EXAMPLES (Refined for sentence length interaction) ---
# Some are kept short, others expanded to 2 sentences (approx 30% expansion)

base_examples = [
    # Identity & Core
    ("Gözlem neden önemlidir?", "Gözlem, bilimsel yöntemlerin temelidir. Doğayı anlamak ve olaylar arasındaki ilişkileri çözmek için gözlem yapmak gerekir."),
    ("İklim değişikliği nedir?", "İklim değişikliği, dünya genelindeki sıcaklık ve hava durumu modellerinde meydana gelen uzun vadeli değişikliklerdir."),
    ("Güneş sistemi hakkında bilgi ver.", "Güneş sistemi, merkezde Güneş ve onun etrafında dönen gezegenler, uydular ve diğer gök cisimlerinden oluşur. Her gezegenin kendine has özellikleri ve yörüngesi vardır."), # Expanded
    ("Su neden önemlidir?", "Su, tüm canlıların yaşamı için hayati öneme sahiptir. Vücut fonksiyonlarının çalışması ve doğanın dengesi, suyun varlığına bağlıdır."), # Expanded
    ("Edebiyatın gücü nedir?", "Edebiyatın gücü, kelimelerle yeni dünyalar yaratabilmesinde yatar. Okuyucuyu farklı zamanlara götürür ve başkalarının duygularını hissetmesini sağlar."), # Expanded
    ("Şiir nedir?", "Şiir, duyguların ve düşüncelerin estetik bir dille ifade edilmesidir. Genellikle ölçülü ve kafiyeli bir yapıya sahiptir."), # Expanded
    ("Roman ile hikaye arasındaki fark nedir?", "Roman genellikle daha uzun, detaylı ve çok karakterli olurken; hikaye daha kısa, öz ve az sayıda olaya odaklanır."),
    ("Ankara nasıl bir yer?", "Ankara, Türkiye'nin başkenti olup, tarihi ve kültürel açıdan zengin bir şehirdir. Cumhuriyet döneminin izlerini taşıyan birçok müzeye ev sahipliği yapar."), # Expanded
    ("Tarih ne işe yarar?", "Tarih, geçmişteki olayları inceleyerek bugünü anlamamıza yardımcı olur. Geçmiş tecrübelerden ders çıkararak geleceğe daha sağlam adımlar atmamızı sağlar."), # Expanded
    ("Kültür nedir?", "Kültür, bir toplumun birikimi, değerleri, gelenekleri, sanatı ve yaşam biçiminin bütünüdür."),
    ("Sağlıklı yaşam için ne yapmalıyım?", "Dengeli beslenmek, düzenli egzersiz yapmak ve yeterli uyumak temel taşlardır. Ayrıca stresten uzak durmak da sağlığı olumlu etkiler."), # Expanded
    ("Kitap okumanın faydaları nelerdir?", "Kitap okumak kelime dağarcığını geliştirir ve hayal gücünü besler. Aynı zamanda empati yeteneğini artırarak farklı bakış açıları kazanmamızı sağlar."), # Expanded
    ("Başarı nedir?", "Başarı, belirlenen hedeflere ulaşmak ve bu süreçte kişisel tatmin sağlamaktır; herkese göre tanımı değişebilir."),
    ("Zaman yönetimi neden önemli?", "Zaman yönetimi, işleri verimli yapmak ve stresi azaltmak için gereklidir. Planlı olmak hayatı kolaylaştırır."),
    ("Öğrenmek zor mu?", "Öğrenmek çaba gerektirir ancak doğru yöntemlerle çok keyifli bir sürece dönüşebilir. Merak duygusunu korumak öğrenmeyi kolaylaştıran en önemli faktördür."), # Expanded
    ("Bana bir hikaye anlat.", "Bir zamanlar, uzak bir ormanda yaşayan meraklı bir tavşan vardı. Her gün yeni yerler keşfetmek isterdi."),
    ("Yapay zeka tehlikeli mi?", "Yapay zeka bir araçtır; nasıl kullanıldığına bağlı olarak faydalı veya zararlı olabilir. Önemli olan etik kurallara uymak ve teknolojiyi insanlık yararına geliştirmektir.") # Expanded
]

daily_examples = [
    ("Sabahları uyanmakta zorlanıyorum, ne yapabilirim?", "Uyku düzenini sabitlemek ve akşamları kafein tüketimini azaltmak yardımcı olabilir. Sabahları odana güneş ışığı girmesini sağlamak da uyanmayı kolaylaştırır."), # Expanded
    ("Stresle nasıl başa çıkabilirim?", "Derin nefes egzersizleri ve kısa yürüyüşler stresi azaltmaya yardımcı olabilir."),
    ("Hangi sporları yapabilirim?", "İlgi alanlarına göre yürüyüş, yüzme, koşu veya yoga gibi sporları deneyebilirsin. Önemli olan sürdürülebilir olmasıdır."),
    ("Meyve yemek neden faydalı?", "Meyveler vitamin, mineral ve lif açısından zengindir. Bağışıklık sistemini güçlendirir ve enerji verir."),
    ("Kahvenin fazlası zararlı mı?", "Evet, aşırı kahve tüketimi uykusuzluk ve çarpıntıya yol açabilir. Bu yüzden günde birkaç fincanla sınırlamak en doğrusudur."), # Expanded
    ("Ders çalışırken odaklanamıyorum.", "Çalışma ortamını sadeleştirmek dikkatin dağılmasını engelleyebilir. Ayrıca kısa molalar vererek çalışmak zihnini taze tutmana yardımcı olur."), # Expanded
    ("Bugün hava nasıl olur?", "Hava durumu sürekli değiştiği için güncel bir hava durumu kaynağından kontrol etmen en doğrusu olacaktır."),
    ("Yemek yapmayı öğrenmek zor mu?", "Hayır, basit tariflerle başlayarak zamanla kendini geliştirebilirsin. Hata yapsan bile denemekten vazgeçmemek önemlidir."), # Expanded
    ("Neden su içmeliyiz?", "Vücudumuzun büyük kısmı sudan oluşur ve organların çalışması için su şarttır. Yeterli su içmek cildin parlamasını sağlar ve enerji verir."), # Expanded
    ("Tasarruf yapmak için ne önerirsin?", "Gereksiz harcamaları not etmek ve bütçe planlaması yapmak tasarruf etmenin ilk adımıdır."),
]

culture_science_examples = [
    ("İstanbul'u özel kılan nedir?", "İstanbul, Asya ve Avrupa kıtalarını birbirine bağlayan eşsiz bir şehirdir. Tarihi yarımadası ve boğazıyla ziyaretçilerine büyüleyici manzaralar sunar."), # Expanded
    ("Kapadokya nerede?", "Kapadokya, Türkiye'nin İç Anadolu Bölgesi'nde, özellikle Nevşehir çevresinde yer alır. Peri bacaları ve balon turlarıyla dünyaca ünlü bir turizm merkezidir."), # Expanded
    ("Çay nasıl demlenir?", "Kaynamış suyu demliğe döküp üzerine çayı ekleyin ve kısık ateşte 15-20 dakika demleyin. İyi bir lezzet için porselen demlik kullanmak önerilir."), # Expanded
    ("Türk kahvesinin özelliği nedir?", "Türk kahvesi, telvesiyle pişirilen ve ikram edilen tek kahve türüdür. Küçük fincanlarda, yanında su ve lokumla sunulması bir gelenektir."), # Expanded
    ("Karadeniz bölgesi nasıldır?", "Karadeniz bölgesi, yemyeşil doğası ve yağışlı iklimiyle bilinir."),
    ("Atatürk kimdir?", "Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır. Çağdaş bir ülke kurmak için birçok devrim gerçekleştirmiştir."), # Expanded
    ("Anıtkabir nerede?", "Anıtkabir, Türkiye'nin başkenti Ankara'da bulunur ve Atatürk'ün anıt mezarıdır."),
    ("Türkiye'nin komşuları kimlerdir?", "Türkiye'nin sınır komşuları Yunanistan, Bulgaristan, Gürcistan, Ermenistan, Azerbaycan, İran, Irak ve Suriye'dir."),
    ("Simit nedir?", "Simit, üzeri susamla kaplı, halka şeklinde bir sokak lezzetidir. Genellikle çay ile birlikte kahvaltılarda tüketilir."), # Expanded
    ("Ege denizi nerede?", "Ege Denizi, Türkiye'nin batısında yer alan ve Akdeniz'in bir uzantısı olan denizdir."),
    ("Gökyüzü neden mavidir?", "Güneş ışınları atmosferdeki gazlara çarparak saçılır ve mavi ışık diğer renklerden daha fazla saçıldığı için gökyüzünü mavi görürüz."),
    ("Ağaçlar neden önemlidir?", "Ağaçlar oksijen üretir, havayı temizler ve toprağı erozyona karşı korur."),
    ("Yerçekimi nedir?", "Yerçekimi, kütlesi olan cisimlerin birbirini çekmesidir. Dünya'nın bizi üzerinde tutmasını sağlayan bu kuvvettir."), # Expanded
    ("Deprem neden olur?", "Yer kabuğundaki levhaların hareketi sonucunda ortaya çıkan enerji dalgalanmaları depreme neden olur."),
    ("Kozmos nedir?", "Kozmos, evrenin düzenli ve ahenkli bütünlüğünü ifade eder. Yıldızlar, gezegenler ve galaksilerin tamamını kapsayan sonsuz bir sistemdir."), # Added
    ("Balıklar nasıl nefes alır?", "Balıklar, suda çözünmüş oksijeni süzebilen solungaçları sayesinde nefes alırlar."),
    ("Gökkuşağı nasıl oluşur?", "Gökkuşağı, güneş ışınlarının yağmur damlalarından geçerken kırılmasıyla oluşur."),
    ("Dünya dönüyor mu?", "Evet, Dünya hem kendi ekseni etrafında hem de Güneş'in etrafında sürekli dönmektedir. Bu hareket gece-gündüz oluşumunu ve mevsimleri sağlar."), # Expanded
]

concepts_misc_examples = [
    ("Mutluluk nedir?", "Mutluluk, kişinin hayatından memnuniyet duyması ve iç huzuru hissetmesidir. Bazen küçük bir an, bazen de bir hedefe ulaşmak mutluluk kaynağı olabilir."), # Expanded
    ("Dostluk ne demektir?", "Dostluk, güven ve sadakate dayanan karşılıksız bir bağdır."),
    ("Sabır neden erdemdir?", "Sabır, zorluklar karşısında sakin kalabilmek ve fevri davranmadan bekleyebilmektir. Çoğu zaman doğru kararlar, sabırlı bir zihinle verilir."), # Expanded
    ("Empati nedir?", "Empati, olaylara başkasının bakış açısıyla bakabilmektir."),
    ("Adalet nedir?", "Adalet, hak ve hukukun gözetilmesi ve herkese eşit davranılmasıdır."),
    ("Cesaret nedir?", "Cesaret, korkuya rağmen doğru olanı yapabilme gücüdür. Korkusuzluk değil, korkuyu yönetebilme becerisidir."), # Expanded
    ("Teknoloji iyi mi kötü mü?", "Teknoloji bir araçtır; insanlığın yararına kullanılırsa iyi, zarar vermek için kullanılırsa kötü sonuçlar doğurabilir."),
    ("Merak duygusu neden önemli?", "Merak, öğrenmenin ve keşfetmenin itici gücüdür. Yeni bilgiler edinmemizi ve kendimizi geliştirmemizi sağlar."), # Expanded
    # Idioms
    ("Damlaya damlaya göl olur ne demek?", "Küçük birikimlerin zamanla büyüyerek büyük değerlere dönüşebileceğini anlatır. Sabırlı olmanın önemini vurgular."), # Expanded
    ("Ayağını yorganına göre uzat ne demek?", "Harcamaların gelire göre yapılması gerektiğini ifade eder."),
    ("Acele işe şeytan karışır ne demek?", "Düşünmeden ve aceleyle yapılan işlerde hata yapma olasılığının yüksek olduğunu belirtir."),
    ("Tatlı dil yılanı deliğinden çıkarır ne demek?", "Güzel sözlerin en inatçı insanları bile ikna edebileceğini anlatır. İletişimde nezaketin gücünü gösterir."), # Expanded
    ("Dost kara günde belli olur anlamı nedir?", "Gerçek dostların sadece iyi günlerde değil, zor zamanlarımızda da yanımızda olduğunu ifade eder."),
]

# --- 2. NEW CATEGORY: "I DON'T KNOW" / UNCERTAINTY (User Request) ---
unknown_examples = [
    ("Gelecekte uçan arabalar ne zaman yaygınlaşacak?", "Bunu kesin olarak bilmek mümkün değil. Teknolojik gelişmeler hızla ilerlese de, bunun ne zaman günlük hayatın parçası olacağını zaman gösterecek."),
    ("Benim adımı biliyor musun?", "Hayır, senin kim olduğunu veya adını bilemem. Gizliliğine saygı duyuyorum."),
    ("Uzaylılar var mı?", "Bu konuda henüz kesin bir kanıt bulunmuyor. Bilim insanları araştırmalarına devam etse de, şu an için net bir cevabımız yok."),
    ("Yarın piyango kime çıkacak?", "Bunu önceden bilmek imkansızdır. Piyango tamamen şansa dayalı bir sistemdir."),
    ("En güzel renk hangisi?", "Güzellik kişiden kişiye değişen bir kavramdır. Herkesin sevdiği ve güzel bulduğu renk farklı olabilir."),
    ("Sence ben iyi bir insan mıyım?", "Seni şahsen tanımadığım için bu konuda bir yorum yapamam. Ancak kendini sorgulaman ve iyi olmaya çalışman değerli bir çabadır."),
    ("Dünyanın sonu ne zaman gelecek?", "Bu konuda bilimsel teoriler olsa da, kesin bir tarih vermek mümkün değildir."),
    ("Hangi takımı tutmalıyım?", "Bu tamamen senin kişisel zevkine kalmış bir karar. Maçları izleyip hangi takımın oyununu sevdiğine göre karar verebilirsin."),
    ("Şu an aklımdan hangi sayıyı tutuyorum?", "Zihnini okuyamam, bu yüzden tuttuğun sayıyı bilemem."),
    ("Hayatımın aşkını ne zaman bulacağım?", "Geleceği görme yeteneğim yok. Bu tür şeyler hayatın akışı içinde, hiç beklemediğin anlarda gerçekleşebilir.")
]

all_data = base_examples + daily_examples + culture_science_examples + concepts_misc_examples + unknown_examples

def create_dataset():
    # Deduplicate based on question text
    seen_questions = set()
    final_list = []
    
    for q, a in all_data:
        if q not in seen_questions:
            final_list.append((q, a))
            seen_questions.add(q)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for q, a in final_list:
            entry = {
                "text": f"<system>{SYSTEM_PROMPT}</system>\n<user>{q}</user>\n<assistant>{a}</assistant>"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(final_list)} items (including {len(unknown_examples)} 'unknown' examples).")
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_dataset()
