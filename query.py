from rag import answer_question

print(" Ekaimapct Chatbot (type 'exit' to quit)\n")
history=[]

while True:
    q=input("You: ")
    if q.lower() in["exit","quit"]:
        break
    ans=answer_question(q,history)
    print("Bot:",ans,"\n")
    history.append((q,ans))