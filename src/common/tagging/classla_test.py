import classla

classla.download('mk')
nlp = classla.Pipeline('mk', processors='tokenize,pos,lemma')

rows = ["Натаму – в поле битолско",
"чемрее врба проклета –",
"под врбата незнаен гроб,",
"в гроб лежи војник непознат."]

for row in rows:
    doc = nlp(row)
    print(doc)
    for token in doc:
        if not token.is_space and not token.is_punct:
            print(token.text.lower(), token.lemma_, token.pos_)

