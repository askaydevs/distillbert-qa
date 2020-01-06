from model import Model

model = Model('model')
context = 'Merchant Marine Circulars issued by the Directorate General of Consular and Maritime Affairs (DGCN) in the format of this Circular. Under the provisions of Law No. 2 of 17 January 1980, some Merchant Marine Circulars will be designated to have the effect and applicability of rules and regulations. Shipowners, operators, agents and masters of merchant vessels should ensure that Merchant Marine Circulars are maintained current and on board the vessels and that the contents of each be properly known to all persons concerned. New Merchant Marine Circulars will be published and distributed periodically as required. Merchant Marine Circulars not modified should be considered to be valid and still in force. Requests for additional copies of the Merchant Marine Circulars should be requested from the Directorate of Consular and Maritime Affairs, New York Representative Office.'
question = 'What are the provisions mentioned?'
answer = model.predict(context, question)

print('Question: ' + question)
print('Answer: ' + answer['answer'])
