from pathlib import Path


DEFAULT_WORDS_PATH = Path(__file__).resolve().parent / "data" / "words_alpha.txt"


WORDS: list[str] = list("""
aachen aachens aardvark aardvarks aback abacus abacuses abaft abandon abandoned
abandoning abandons abase abased abasement abases abash abashed abashes abashing
abate abated abatement abates abating abbey abbeys abbot abbots abbreviate
abbreviated abbreviation abdomen abdominal abduct abducted abduction able abler
ablest aboard abode abolish abolished abolishes abolition abound abounded about
above abridge abridged abroad abrupt abruptly abscess absence absent absolute
absorb absorbed absorbing absorbs abstract absurd abundance abundant abuse abused
academic academy accelerate accelerated accent accept accepted access accident
account accurate accuse accused ache achieved acid acoustic acquire acre across
action active actor actual adapt added address adjacent adjust admire admit adopt
adult advance advice advise aerobic affair affect afford afraid after again agent
agree agreed ahead alarm album alert alike alive allow almost alone along alter
always amaze amused analog anchor ancient angle angry animal annual answer apart
apology appeal appear apple apply arcade archive area argue arise armed around
array arrest arrive arrow artist aspect asset assist assume atlas atom attack
attend audio author autumn avenue average awake aware awful awkward baker banana
banker banner barrel basic basket batch battery beach beacon beard beauty become
bedroom before begin behave behind belief belong beneath beside better between
beyond bicycle bitter blade blanket blast blend bless blind block bloom board
border borrow bottle bottom branch brave bread bridge bright broken brother
budget build bundle burden button cable camera campus cancel candle cannon canvas
carbon career carpet castle casual catalog catch cause cedar center cereal chain
chair chance change charge charm cheap cheese cherry chest chicken choice choose
circle citizen civil claim classic clean clear clever client clinic clock close
cloud coach coast coffee column comet common copper corner cotton couple cousin
create credit creek crown current custom cycle daily damage danger daring dealer
debate decade decide deep defend degree demand depend desert design detail detect
device dinner direct doctor domain double dozen dragon dream driven during eagle
early earth editor effect effort either elbow elder electric elegant element
energy engine enough entire equal escape estate event every exact example expert
fabric factor fairly family famous farmer fasten father feather fellow figure
filter finder finger finish flower follow forest forget formal format fortune
garden gentle golden guitar handle harbor honest income island jacket jungle
kitten laptop ladder letter market mirror mother motion narrow office orange
paper parade parent pillow planet pocket rabbit random river rocket silver
simple sister summer table talent target tender travel window winter writer
yellow zipper
""".split())


def load_words(path: str | Path | None = None, limit: int | None = None) -> list[str]:
    word_path = Path(path) if path else DEFAULT_WORDS_PATH
    if word_path.exists():
        with word_path.open(encoding="utf-8", errors="ignore") as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        words = WORDS
    return words[:limit] if limit is not None else words
