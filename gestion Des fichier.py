
fichier= open("SouissiHend.txt",'w')
fichier.write("L' intelligence artificielle (IA) est l'ensemble des théories et des techniques mises en œuvre en vue de réaliser des machines capables de simuler l'intelligence. \n Elle correspond donc à un ensemble de concepts et de technologies plus qu'à une discipline autonome constituée")
fichier.close()

fichier = open("SouissiHend.txt")
b=fichier.read()
print(b)

#♦------q3----------#
fichier2=open("SouissiHend2.txt",'w')
fichier2.write("Machine Learning et Deep Learning sont deux modules de l'intelligence artificielle.")
fichier2.close()
fichier2 = open ("SouissiHend2.txt")
while True :
    chaine =fichier2.read(16)
    print (chaine)
    if chaine =="": 
        break
fichier.close ()
#-------q4---------------#
fichier=open("SouissiHend2.txt","a")
fichier.write("\n L' intelligence artificielle (IA) est l'ensemble des théories et des techniques mises en œuvre en vue de réaliser des machines capables de simuler l'intelligence. \n Elle correspond donc à un ensemble de concepts et de technologies plus qu'à une discipline autonome constituée")
fichier.close()

