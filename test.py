def get_data(path):
        #file = pd.read_csv(path)
        #questions = file['questions'].values
        #answers = file['answers'].values

        file = open(path, 'r').read()
        lists = [f.split('\t') for f in file.split('\n')]
        print("lists:\n")
        print(lists)
        print("\n")

        questions = [x[0] for x in lists]
        print("questions:\n")
        print(questions)
        answers = [x[1] for x in lists]
        print("\n")
        print("answers")
        print(answers)



get_data("C:/Users/neilx/Downloads/chatter_bot_3/chatter_bot_3/new1.txt")