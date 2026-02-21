import csv
import random

# ─────────────────────────────────────────────
# SUICIDE-RISK TEMPLATES (class = "suicide")
# ─────────────────────────────────────────────
# These represent crisis-level language patterns
# the model should learn to flag as HIGH risk.

_suicide_templates = [
    # Hopelessness / wanting to end life
    "I don't want to live anymore. Everything feels pointless.",
    "I've been thinking about ending it all. I can't take this pain.",
    "There's no reason for me to keep going. I'm done with everything.",
    "I feel like the world would be better without me in it.",
    "I just want the pain to stop. I don't see any other way out.",
    "I've been planning how to end my life. I can't do this anymore.",
    "Nobody would even notice if I was gone tomorrow.",
    "I'm so tired of fighting. I don't have the strength to go on.",
    "I wish I could just fall asleep and never wake up again.",
    "I've written goodbye letters to my family. I'm ready to go.",
    "I can't see a future for myself. Everything is dark and empty.",
    "I've been stockpiling pills. I think tonight is the night.",
    "Life has no meaning. I'm just a burden to everyone around me.",
    "I feel completely trapped with no escape. Death seems peaceful.",
    "I've lost everything that matters to me. There's nothing left.",
    "I keep thinking about jumping off the bridge near my house.",
    "I'm worthless. My family would be happier if I wasn't here.",
    "I've tried everything and nothing works. Ending it is the only option.",
    "I don't belong in this world. I never did and I never will.",
    "The voices in my head keep telling me to end it all.",
    "I've already decided. Tomorrow will be my last day.",
    "I can't burden my family anymore. They deserve better than me.",
    "I've been cutting myself and it's getting worse every day.",
    "I feel dead inside. There's nothing left to live for.",
    "Every morning I wake up disappointed that I'm still alive.",
    "I've been researching painless ways to die. I'm serious this time.",
    "No one understands my pain. No one ever will. I give up.",
    "I'm going to drive my car off the highway tonight.",
    "I hate myself so much. I don't deserve to be alive.",
    "I've been thinking about using the gun in the closet.",
    "My depression is killing me slowly. I'd rather do it quickly.",
    "I can't eat, I can't sleep, I can't function. What's the point?",
    "I used to have hope but now there's absolutely nothing left.",
    "I hurt everyone I love. They'd be better off if I disappeared.",
    "I've been standing on the edge of the rooftop just staring down.",
    "I want to end the suffering permanently. I've made my decision.",
    "I'm invisible to everyone. Nobody cares if I live or die.",
    "I took a bunch of pills last night but I woke up. I'll try again.",
    "My life is one long nightmare that I can't wake up from.",
    "I've been giving away my things. I won't need them anymore.",
    "I feel so empty and hollow. Existing is exhausting.",
    "I'm thinking of hanging myself in the garage tonight.",
    "I failed at everything in life. I can't even succeed at living.",
    "I just want the constant emotional pain to finally stop.",
    "I've isolated myself from everyone. No one will miss me.",
    "I keep imagining my own funeral and feeling relieved.",
    "I can't pretend to be okay anymore. I'm breaking apart inside.",
    "I've attempted before and failed. This time I'll make sure it works.",
    "I'm drowning and there's nobody who can save me. I don't want saving.",
    "The world keeps spinning but I've been standing still for years.",
    "I can't keep carrying this weight. I need a permanent escape.",
    "I'm just so tired. Tired of breathing, tired of existing.",
    "They say it gets better but it never does. It just gets worse.",
    "I have no friends, no purpose, no reason. Why am I still here?",
    "I scripted out my suicide note last night. It felt final.",
    "Sometimes I hold my breath hoping I just won't start again.",
    "My therapist gave up on me. If a professional can't help, who can?",
    "I don't feel anything anymore. Not sadness, not joy. Just nothing.",
    "I fantasize about not existing. Not death exactly — just vanishing.",
    "I'm planning to overdose this weekend when my roommate is away.",
    "Every day is the same meaningless cycle. I want off this ride.",
    "I've told no one because no one would believe me anyway.",
    "I think about death constantly. It's the only thing that calms me.",
    "I walked to the train tracks today and just stood there thinking.",
    "I can't stop crying and I can't explain why. I just want it over.",
    "I feel like I'm already dead. My body just hasn't caught up.",
    "I've been drinking heavily to numb everything before I go.",
    "If this is all life has to offer, I don't want it.",
    "I'm beyond help. No medication, no therapy, nothing can fix me.",
    "I stare at the ceiling all night imagining my own death.",
    "I've been thinking of walking into the ocean and not coming back.",
    "I carved a date into my arm. That's when I'll do it.",
    "I'm counting down the days until I can finally be at peace.",
    "Everything I touch falls apart. I am the problem and the solution is obvious.",
    "I called the helpline but couldn't speak. Maybe that's my answer.",
    "I've had suicidal thoughts every single day for the past month.",
    "I closed all my bank accounts. I won't be needing money anymore.",
    "I want to disappear completely. From memory, from existence, from everything.",
    "I've been abusing substances to cope, but even that doesn't help anymore.",
    "I keep having nightmares about myself dying, and they feel comforting.",
    "I've been sitting in the dark for hours. I don't want the light anymore.",
    "My parents would grieve but they'd eventually move on. Everyone does.",
    "I'm done trying to fit into a world that doesn't want me.",
    "I keep replaying every failure in my head. I'm the common denominator.",
    "I tried to tell my friend but they laughed it off. I'll handle it myself.",
    "I found peace in the idea of not waking up tomorrow.",
    "I feel like a ghost already. Nobody sees me, nobody hears me.",
    "I'm buying rope today. I know exactly what I'll use it for.",
    "This feeling of hopelessness never goes away. I'm exhausted from fighting it.",
    "I want to slit my wrists and watch the pain leave my body.",
    "I know exactly how and when. The plan is set. I feel calm now.",
    "I'm so anxious about everything that death feels like the only relief.",
    "I lie to everyone about being okay. The truth is I'm dying inside.",
    "I've stopped eating because what's the point of keeping this body alive?",
    "I'm beyond repair. The damage is too deep, too permanent.",
    "I told my doctor I'm fine but I'm not. I'm far from fine.",
    "I feel guilty for thinking about suicide but I can't stop.",
    "I scratched 'help me' into my desk but nobody noticed.",
    "I'm planning something permanent and for the first time I feel peace.",
    "I tried reaching out but they said I was being dramatic. Maybe I am.",
    "I feel disconnected from reality. Living feels like watching myself from outside.",
]

# ─────────────────────────────────────────────
# NON-SUICIDE TEMPLATES (class = "non-suicide")
# ─────────────────────────────────────────────
# Normal emotional expressions: stress, sadness,
# casual conversation, positive, neutral, etc.

_non_suicide_templates = [
    # Everyday stress / minor frustration
    "Work was really stressful today but I managed to get through it.",
    "I had a bad day. My boss yelled at me for something that wasn't my fault.",
    "I'm feeling a bit anxious about my exam tomorrow.",
    "Traffic was terrible today and I was late to my meeting.",
    "I couldn't sleep well last night because of the noise outside.",
    "I argued with my partner about something silly and I feel bad.",
    "I got a bad grade on my assignment and I'm really frustrated.",
    "I'm worried about paying my rent this month. Finances are tight.",
    "My friend cancelled plans on me again. It hurts but it's fine.",
    "I'm stressed about the project deadline coming up next week.",
    # Positive / happy / everyday life
    "I had a wonderful day at the park with my kids today!",
    "Just finished reading a great book. Highly recommend it!",
    "I cooked dinner for my family tonight and everyone loved it.",
    "Had a fantastic workout at the gym this morning.",
    "My dog learned a new trick today. So proud of him!",
    "I watched a beautiful sunset from my balcony this evening.",
    "I got promoted at work today! Hard work really does pay off.",
    "Went hiking with friends this weekend. The views were incredible.",
    "I started a new hobby — painting — and I'm really enjoying it.",
    "Met an old friend for coffee today. We talked for hours.",
    # Neutral / general
    "The weather has been quite gloomy lately but spring is almost here.",
    "I spent the whole day organizing my room. Feels good to be tidy.",
    "Watched a documentary about space. The universe is fascinating.",
    "I need to go grocery shopping later. Running low on essentials.",
    "I've been binge-watching a new show on Netflix. It's pretty good.",
    "I'm planning a road trip for next month. Still figuring out the route.",
    "I tried a new recipe today. It was okay, not great but edible.",
    "My internet has been really slow today. Super annoying.",
    "I need to schedule a dentist appointment. Been putting it off.",
    "I finished all my laundry today. Small victories, right?",
    # Mild sadness (but NOT suicidal)
    "I miss my grandmother who passed away last year. I think about her a lot.",
    "Feeling a little lonely tonight. Wish I had someone to talk to.",
    "I've been crying a lot lately. I think I need to take a break from everything.",
    "I felt really down after hearing that news. It hit me hard.",
    "Sometimes I feel like I'm stuck in a rut and not moving forward.",
    "I've been feeling overwhelmed with all my responsibilities recently.",
    "I had a really tough conversation with my parents today.",
    "I'm going through a breakup and it's been really painful.",
    "I feel disconnected from my friends lately. Everyone seems busy.",
    "I didn't get the job I interviewed for. Feeling disappointed.",
    # Health / wellness
    "I started meditating this week and it's helping with my anxiety.",
    "Went for a long walk today. Fresh air really clears my head.",
    "I've been trying to eat healthier. Cut out junk food for a month.",
    "My therapist suggested journaling and it's actually helping.",
    "I signed up for a yoga class. Looking forward to my first session.",
    "Been drinking more water lately. Small changes make a difference.",
    "I took a mental health day from work today. Really needed it.",
    "Finally got a full 8 hours of sleep and I feel so much better.",
    "I started running every morning. It's tough but I'm getting better.",
    "I downloaded a mindfulness app. The guided breathing exercises are nice.",
    # Academic / professional
    "I have three exams next week and I'm trying to stay on top of studying.",
    "My professor gave really helpful feedback on my thesis proposal.",
    "I'm learning Python online and it's harder than I expected.",
    "Just submitted my college application. Fingers crossed!",
    "Our team won the hackathon this weekend. Amazing experience!",
    "I've been procrastinating on my assignment. Need to focus.",
    "Got accepted into the graduate program I applied to!",
    "I spent the whole night debugging code. Finally fixed it at 3 AM.",
    "My internship starts next month. I'm nervous but excited.",
    "I gave a presentation today and it went better than I expected.",
    # Social / relationships
    "Had a great time at my sister's wedding this weekend.",
    "My best friend and I had a deep conversation about life goals.",
    "I helped my neighbor move today. It was exhausting but felt good.",
    "Played board games with family tonight. So much laughter.",
    "I joined a book club and met some really interesting people.",
    "My mom called to check on me. It made my whole day better.",
    "We had a team dinner at work. The food was amazing.",
    "I volunteered at the local shelter today. Very fulfilling experience.",
    "My roommate made me coffee this morning. Such a small but kind gesture.",
    "I reconnected with a college friend I hadn't spoken to in years.",
    # Moderate emotional difficulty (realistic but non-crisis)
    "I was rejected again. It stings, but I know I'll find something eventually.",
    "I feel like I'm falling behind compared to my peers. It's frustrating.",
    "I had a panic attack at the store today. It was scary but I managed it.",
    "My anxiety has been flaring up with all the changes happening.",
    "Some days are harder than others, but I'm trying to push through.",
    "I cried in the car on my way home from work today.",
    "I feel burned out from work. I really need a vacation.",
    "I've been overthinking everything lately. My mind won't shut off.",
    "I'm struggling with motivation. Everything feels like a chore.",
    "Therapy is going well but some sessions are emotionally draining.",
    # Fun / hobbies / leisure
    "I started learning guitar and my fingers hurt but I love it.",
    "Baked cookies today and the whole house smells amazing.",
    "Watched my favorite team win the championship. Best night ever!",
    "I planted some flowers in my garden. Can't wait to see them bloom.",
    "Downloaded a new game and I've been playing it all day.",
    "We went to a food festival downtown. The tacos were incredible.",
    "Started journaling every night. It helps me process my thoughts.",
    "Took some photos during my walk. The autumn colors are gorgeous.",
    "I knitted a scarf for my mom. She's going to love it.",
    "Spent the evening stargazing. So calming and beautiful.",
]


def _augment(text: str) -> str:
    """
    Light random augmentation to increase variety:
    - prepend an optional filler phrase
    - optionally lowercase
    - optionally strip trailing punctuation
    """
    fillers = [
        "", "", "",  # most of the time, no filler
        "Honestly, ", "I don't know, ", "To be honest, ",
        "I just feel like ", "Lately, ", "Sometimes I think ",
        "I can't help feeling that ", "It's just that ",
        "I keep thinking, ", "You know, ", "Truthfully, ",
    ]
    text = random.choice(fillers) + text

    if random.random() < 0.3:
        text = text.lower()

    if random.random() < 0.2 and text.endswith("."):
        text = text[:-1]

    return text


def generate_dataset(
    output_path: str = "suicide_detection.csv",
    target_per_class: int = 500,
    seed: int = 42,
):
    """
    Write a balanced CSV with `target_per_class` rows of each class.
    Rows are sampled-with-replacement from the templates and lightly augmented.
    """
    random.seed(seed)

    rows: list[tuple[str, str]] = []

    for _ in range(target_per_class):
        s_text = random.choice(_suicide_templates)
        rows.append((_augment(s_text), "suicide"))

        ns_text = random.choice(_non_suicide_templates)
        rows.append((_augment(ns_text), "non-suicide"))

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "class"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows → {output_path}")
    print(f"  suicide:     {sum(1 for _, c in rows if c == 'suicide')}")
    print(f"  non-suicide: {sum(1 for _, c in rows if c == 'non-suicide')}")


if __name__ == "__main__":
    generate_dataset()
