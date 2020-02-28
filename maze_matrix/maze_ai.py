""" definirea problemei """
import copy
import math

import configuratii as cf


class Nod:
    NR_LINII = 20
    NR_COLOANE = 20

    def __init__(self, info, h):
        self.info = info
        self.h = h

    def __str__(self):
        sir = '\n'
        for ind, i in enumerate(self.info):
            sir += '\n'
            for indj, j in enumerate(i):
                sir += ' ' + str(j)
        return sir

    def __repr__(self):
        sir = '\n'
        sir += str(self.info)
        return sir


class Graf:

    def __init__(self, pozitii, scop):
        self.nod_start = Nod(list(pozitii), float('inf'))
        self.reprez_scop = list(scop)

    def scop(self, nod):
        return nod.info == self.reprez_scop

    def calculeaza_h(self, reprez_nod):
        # distanta euclidiana intre locatia lui 1 in reprez_nod
        # si locatia lui 1 in reprez_scop
        for i in range(Nod.NR_LINII):
            for j in range(Nod.NR_COLOANE):
                if reprez_nod[i][j] == 1:
                    i_nod = i
                    j_nod = j
        for i in range(Nod.NR_LINII):
            for j in range(Nod.NR_COLOANE):
                if self.reprez_scop[i][j] == 1:
                    i_scop = i
                    j_scop = j
        return math.sqrt((i_scop - i_nod) ** 2 + (j_scop - j_nod) ** 2)

    def interschimba(self, i_vechi, j_vechi, i_nou, j_nou, matr):
        temp = matr[i_nou][j_nou]
        matr[i_nou][j_nou] = matr[i_vechi][j_vechi]
        matr[i_vechi][j_vechi] = temp
        return matr

    def calculeaza_succesori(self, nod):
        l_succesori = []

        for i in range(nod.NR_LINII):
            for j in range(nod.NR_COLOANE):
                if nod.info[i][j] == 1:
                    i_poz = i
                    j_poz = j

        if i_poz > 0:
            i_nou = i_poz - 1
            i_vechi = i_poz
            j_nou = j_vechi = j_poz
            matr_noua = copy.deepcopy(nod.info)
            matr_noua = self.interschimba(i_vechi, j_vechi, i_nou, j_nou, matr_noua)
            h_nou = self.calculeaza_h(matr_noua)
            l_succesori.append((Nod(matr_noua, h_nou), 1))
        if i_poz < nod.NR_LINII - 1:
            i_nou = i_poz + 1
            i_vechi = i_poz
            j_nou = j_vechi = j_poz
            matr_noua = copy.deepcopy(nod.info)
            matr_noua = self.interschimba(i_vechi, j_vechi, i_nou, j_nou, matr_noua)
            h_nou = self.calculeaza_h(matr_noua)
            l_succesori.append((Nod(matr_noua, h_nou), 1))
        if j_poz > 0:
            j_nou = j_poz - 1
            j_vechi = j_poz
            i_nou = i_vechi = i_poz
            matr_noua = copy.deepcopy(nod.info)
            matr_noua = self.interschimba(i_vechi, j_vechi, i_nou, j_nou, matr_noua)
            h_nou = self.calculeaza_h(matr_noua)
            l_succesori.append((Nod(matr_noua, h_nou), 1))
        if j_poz < nod.NR_COLOANE - 1:
            j_nou = j_poz + 1
            j_vechi = j_poz
            i_nou = i_vechi = i_poz
            matr_noua = copy.deepcopy(nod.info)
            matr_noua = self.interschimba(i_vechi, j_vechi, i_nou, j_nou, matr_noua)
            h_nou = self.calculeaza_h(matr_noua)
            l_succesori.append((Nod(matr_noua, h_nou), 1))

        return l_succesori


""" Sfarsit definire problema """

""" Clase folosite in algoritmul A* """


class NodCautare:
    def __init__(self, nod_graf, succesori=[], parinte=None, g=0, f=None):
        self.nod_graf = nod_graf
        self.succesori = succesori
        self.parinte = parinte
        self.g = g
        if f is None:
            self.f = self.g + self.nod_graf.h
        else:
            self.f = f

    def drum_arbore(self):
        nod_c = self
        drum = [nod_c]
        while nod_c.parinte is not None:
            drum = [nod_c.parinte] + drum
            nod_c = nod_c.parinte
        return drum

    def contine_in_drum(self, nod):
        nod_c = self
        while nod_c.parinte is not None:
            if nod.info == nod_c.nod_graf.info:
                return True
            nod_c = nod_c.parinte
        return False

    def __str__(self):
        parinte = self.parinte if self.parinte is None else self.parinte.nod_graf.info
        # return "("+str(self.nod_graf)+", parinte="+", f="+str(self.f)+", g="+str(self.g)+")";
        return str(self.nod_graf)


""" Algoritmul A* """


def debug_str_l_noduri(l):
    sir = ""
    for x in l:
        sir += str(x) + "\n"

    return sir


def get_lista_solutii(l):
    drum = []
    for x in l:
        drum.append(x.nod_graf.info)
    return drum


def in_lista(l, nod):
    for x in l:
        if x.nod_graf.info == nod.info:
            return x
    return None


def a_star(graf):
    rad_arbore = NodCautare(nod_graf=graf.nod_start)
    print(graf.nod_start.info)
    print(graf.nod_start)

    open_list = [rad_arbore]
    closed = []
    ok = False
    while len(open_list) > 0:
        nod_curent = open_list.pop(0)
        closed.append(nod_curent)
        if graf.scop(nod_curent.nod_graf):
            ok = True
            break
        l_succesori = graf.calculeaza_succesori(nod_curent.nod_graf)
        for (nod, cost) in l_succesori:
            if not nod_curent.contine_in_drum(nod):
                x = in_lista(closed, nod)
                g_succesor = nod_curent.g + cost
                f = g_succesor + nod.h
                if x is not None:
                    if f < nod_curent.f:
                        x.parinte = nod_curent
                        x.g = g_succesor
                        x.f = f
                else:
                    x = in_lista(open_list, nod)
                    if x is not None:
                        if x.g > g_succesor:
                            x.parinte = nod_curent
                            x.g = g_succesor
                            x.f = f
                    else:  # cand nu e nici in closed nici in open
                        nod_cautare = NodCautare(nod_graf=nod, parinte=nod_curent, g=g_succesor)
                        open_list.append(nod_cautare)
        open_list.sort(key=lambda x: (x.f, x.g))

    if ok == True:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        # puneti o cale valida
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        #f = open('D:/PythonWorkspace/maze/demo_file.txt', 'a')
        #f.write('---------------New run from here------------------')
        #f.write(debug_str_l_noduri(nod_curent.drum_arbore()))
        print(debug_str_l_noduri(nod_curent.drum_arbore()))
        return get_lista_solutii(nod_curent.drum_arbore())
    else:
        return []


def main(pozitii, scop):
    problema = Graf(pozitii, scop)
    return a_star(problema)


main(cf.start, cf.scop)
