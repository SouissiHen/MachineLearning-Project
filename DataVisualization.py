import matplotlib.pyplot as plt
plt.figure()
Nb=[10,20,30,40,50,60,70,80,90,100]
T=[2,4,6,8,10,12,14,16,18,20]
plt.plot(T,Nb)
plt.xlabel('temps')
plt.ylabel('Piece conforme')
plt.title('Courbe représentative du pièces conformes fabriquées en fonction du temps')
plt.axis([8,16,40,80])

T=[10,12,14,16,18,20]
NB1=[10,20,30,40,50,60]
T=[10,12,14,16,18,20]
NB2=[30,40,50,60,70,80]
T=[10,12,14,16,18,20]
NB3=[40,50,60,70,80,90]
plt.plot(T,NB1,'r--')
plt.plot(T,NB2,'b*')
plt.plot(T,NB3,'g+')
plt.plot(T,NB1,'r--' , linewidth=7 , label='Fabrication1')
plt.legend()
plt.plot(T,NB2,'b*', linewidth=12 , label='Fabrication2')
plt.legend()
plt.plot(T,NB3,'g+', linewidth=10 , label='Fabrication3')
plt.legend()


plt.text(16,40,'Nombre optimal',horizontalalignment='center',verticalalignment='center' )

plt.figure()
Catégorie = ['Catégorie1', 'Catégorie2', 'Catégorie3', 'Catégorie4']
Nombre = [5000, 26000, 21400, 12000]
plt.pie(Nombre,explode=(0.15,0,0,0),labels=Catégorie)





