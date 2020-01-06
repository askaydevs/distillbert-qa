from model import Model

model = Model('model')

context = "World War II (often abbreviated to WWII or WW2), also known as the Second World War, \
 was a global war that lasted from 1939 to 1945. The vast majority of the world's countries—including all \
 the great powers—eventually formed two opposing military alliances: the Allies and the Axis. A state of total war emerged, \
 directly involving more than 100 million people from more than 30 countries. The major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, marked by 70 to 85 million fatalities, most of whom were civilians in the Soviet Union and China. It included massacres, the genocide of the Holocaust, strategic bombing, premeditated death from starvation and disease, and the only use of nuclear weapons in war."

question = "What years did WW2 last between?"

answer = model.predict(context, question)

print("Question: " + question)
print("Answer: " + answer["answer"])
