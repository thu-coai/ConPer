class RelationList:
    def __init__(self):
        self.relation2id = {}
        self.cnt = 0
    
    def add(self, relation):
        if relation not in self.relation2id:
            self.relation2id[relation] = self.cnt
            self.cnt += 1
        return self.relation2id[relation]

    def get_idx(self, relation):
        return self.relation2id[relation]

    def __len__(self):
        return self.cnt
    
    def __repr__(self):
        result = ''
        for k, v in self.relation2id.items():
            result += f'{k} : {v}\n'
        return result


class KnowledgeGraph:

    def __init__(self, edges, bidir=False):
        self.data = {}
        self.prev = {}
        self.weights = {}
        self.relations = RelationList()

        self.relations.add('unrelated') # relation_id 0 对应和NOT_A_FACT的连接

        for item in edges:
            # [head, relation, tail, weight]
            head = item[0]
            relation = item[1]
            tail = item[2]

            if not self.eng_word(head) or not self.eng_word(tail):
                continue
            assert '/' not in head and '/' not in tail

            relation_id = self.relations.add(relation)
            self.add(head, relation_id, tail)
            if bidir:
                self.add(tail, relation_id, head)
            self.weights[self.get_name(item[0], item[-2])] = float(item[-1])
            if bidir:
                self.weights[self.get_name(item[-2], item[0])] = float(item[-1])

        print(f"relation nums:{len(self.relations)}")
        print(self.relations)
    
    def get_relation_size(self):
        # 返回relation总数
        return len(self.relations)
        
    def get_relation_list(self):
        return self.relations

    # def get_relation_idx(self, relation):
    # 加进图谱的时候已经转换成id了
    #     return self.relations.get_idx(relation)

    def filter_points(self, points):
        res = []
        for pt in points:
            if pt in self.data:
                res.append(pt)
        return res

    def check(self, point):
        return point in self.data

    def get_name(self, src, dst):
        return src + "___" + dst

    def get_weight(self, src, dst):
        name = self.get_name(src, dst)
        if name in self.weights:
            return self.weights[name]
        return None

    def eng_word(self, word):
        if '_' in word:
            return False
        return True

    def get_avg_deg(self):
        r = 0
        for src in self.data:
            r += len(self.data[src])

        return r / len(self.data)

    def show_degs(self):
        data = list(self.data.items())
        print(data[-3:])
        data.sort(key=lambda x: len(x[1]))
        for k, v in data:
            print(f'{k}:{len(v)}')

    def get_node_num(self):
        return len(self.data)

    def add(self, src, relation, dst):
        w = (dst, relation)
        if src in self.data:
            if w not in self.data[src]:
                self.data[src].append(w)
        else:
            self.data[src] = [w]

        q = (src, relation)
        if dst in self.prev:
            if q not in self.prev[dst]:
                self.prev[dst].append(q)
        else:
            self.prev[dst] = [q]


    def get_neighbors(self, pt, relation=False):
        if pt not in self.data:
            return []
        else:
            if relation:
                return self.data[pt]
            else:
                return [i[0] for i in self.data[pt]]


    def get_triples(self, word):

        res = []
        if word in self.data:
            for dst, r in self.data[word]:
                res.append((word, r, dst))

        if word in self.prev:
            for src, r in self.prev[word]:
                res.append((src, r, word))
        
        if not res:
            res.append((word, 0, 'NOT_A_FACT'))

        return res

    def get_hops_set(self, srcs, hop, relation=False):
        res = set(srcs)
        step = 0
        temp = set(srcs)
        while step < hop:
            step += 1
            new_temp = []
            for pt in temp:
                ns = self.get_neighbors(pt, relation=relation)
                for n in ns:
                    if n not in res:
                        new_temp.append(n)
            new_temp = set(new_temp)
            temp = new_temp
            res = res | new_temp
        return res
        
    def get_intersect(self, srcs, dsts, hop=2):
        src_neis = self.get_hops_set(srcs, hop)
        dst_neis = self.get_hops_set(dsts, hop)
        return src_neis & dst_neis

    def find_neigh_in_set(self, src, points):
        res = []
        if src not in self.data:
            return res
        for pt in points:
            if pt in self.data[src]:
                res.append(pt)
        return set(res)

    def find_paths(self, srcs, dsts):
        a = self.get_hops_set(srcs, 1)
        res = []
        for w in a:
            x = self.find_neigh_in_set(w, srcs)
            y = self.find_neigh_in_set(w, dsts)
            if x and y:
                res.append([x, w, y])
        return res

    def show_paths(self, srcs, dsts):
        paths = self.find_paths(srcs, dsts)
        for path in paths:
            print(path)

    def get_dis(self, dst, srcs, max_hop=3):
        vis = set()
        points = [dst]
        vis.add(dst)
        step = 0
        if dst in srcs:
            return step
        while step < max_hop:
            step += 1
            temp_points = []
            for pt in points:
                ns = self.get_neighbors(pt)
                for n in ns:
                    if n in srcs:
                        return step
                    if n in vis:
                        continue
                    vis.add(n)
                    temp_points.append(n)
            points = temp_points
        return step


def get_conceptnet(path):

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        print(len(lines))
        edges = []
        for line in lines:
            edge = line.strip().split('|||')
            edges.append(edge)

        return KnowledgeGraph(edges)


if __name__ == '__main__':

    graph = get_conceptnet()
    print(f"node num:{graph.get_node_num()}, avg deg:{graph.get_avg_deg()}")
   
    print(graph.get_hops_set(['people'], hop=1, relation=False))
    print('='*100)
    print(graph.get_hops_set(['people'], hop=1, relation=True))
    # graph.show_degs()
    # print(graph.get_hops_set(['people'], 1))