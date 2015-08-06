import os
import wx
import datetime

nodesList=[]
nodenamesList=[]
networkList=[]
evidenceList=[]

# -*- coding: utf-8 -*-

"""

Bayesian Network Assignment 2


The objective of this assignment is to develop a Bayesian Network module with

the following functionalities:

1. Creating a node

2. Setting the Conditional Probability Distribution of the node

3. Initializing a Bayesian network

4. Adding/Deleting a node to/from the Bayesian network

5. Displaying the structure of the Bayesian network

6. Visualizing the Bayesian network graphically

7. Getting list of children/parents from a given node

8. Save/Load the Bayesian network to/from human readable json format

9. Set/remove hard evidence on nodes

10. Bayesian inference to compute posterior probabilities given hard evidence

11. Marginal probability distribution of individual nodes



"""



import json

import networkx as nx

import matplotlib.pylab as plt

def extend(d, k, v):

    """Returns a new dict containing elements of d and the new key,value pair

    indicated by k,v"""

    n = d.copy()

    n[k] = v

    return n

def cut(d, k):

    """Returns a new dict containing elements of d, excepting the key,value

    pair with key k (if it exists in d).

    If d is a sequence, returns a new list containing all elements of d not

    equal to k."""

    if isinstance(d, dict):

        n = d.copy()

        if k in n:

            del n[k]

        return n

    return [v for v in d if v != k]

def reset(bn):

    """This method takes a Bayesian network object as parameter and  resets 

    a Bayesian network to its original configuration with no nodes"""

    bn.nodes=[]

    
def display(bn):

    """This method displays the Bayesian network as a directed graph with 

    directed edge from the parent node to child node"""

    fig=plt.figure()

    g=nx.DiGraph()

    structure=bn.displayStructure()

    for node in bn.nodes:

        nodename=node.name

        g.add_node(nodename)

    for x in g.nodes():

        if len(structure[x])!=0:

            for y in structure[x]:

                g.add_edge(y,x)

    nx.draw(g)

    return fig







def json_serialize(obj, filename, use_jsonpickle=True):

    """This method take the Bayesian network object and json filename as 

    arguments and saves the Bayesian network object in a human readable 

    json format"""

    f = open(filename, 'w')

    if use_jsonpickle:

        import jsonpickle

        json_obj = jsonpickle.encode(obj)

        f.write(json_obj)

    else:

        json.dump(obj, f) 

    f.close()

    



def json_load_file(filename, use_jsonpickle=True):

    """This method takes the json filename as argument and returns a 

    Bayesian network object by loading it from the json file"""

    try:

        f = open(filename)

        if use_jsonpickle:

            import jsonpickle

            json_str = f.read()

            obj = jsonpickle.decode(json_str)

        else:

            obj = json.load(f)

        return obj

    except IOError:

        print "Problem reading file ",filename

         

def normalize(dist):

    """Normalizes a probability distribution so it sums to 1.  The distribution

    may be a list or a dictionary of value,probability pairs.    

    This function modifies the original object."""

    

    if isinstance(dist, dict):

    # Make sure our keys/values line up in their lists

        keys = dist.keys()

        vals = [dist[k] for k in keys]

        normalize(vals)

        for k,v in zip(keys,vals):

            dist[k] = v

        return

    fdist = [float(d) for d in dist]

    s = sum(fdist)

    if s == 0:

            return

    fdist = [d/s for d in fdist]

    for i,d in enumerate(fdist):

        dist[i] = d





class DiscreteCPT(object):

    """The conditional probability distribution of a give node in the Bayesian

    nrtwork"""

    def __init__(self, vals, probTable):

        self.myVals = vals

        if isinstance(probTable, list) or isinstance(probTable, tuple):

                self.probTable = {(): probTable}

        else:

                self.probTable = probTable

  

    def values(self):

        """This method returns the values corresponding to the conditional

        probability distribution of the node."""

        return self.myVals

    

  

        

    def prob_dist(self, parentVals):

        """Returns a dictionary giving a value:probability mapping for each

        value the variable can assume, given a tuple with values for all the

        conditions.

        """

        if isinstance(parentVals, list):

            parentVals = tuple(parentVals)

        return dict([(self.myVals[i],p) for i,p in \

                    enumerate(self.probTable[parentVals])]) 

    

class DiscreteBayesNode(object):

    """A node in a Bayesian Network of discrete valued variables."""

    def __init__(self,name,parents):

        self.name=name

        self.parents=parents

        self.cpt = DiscreteCPT([],{})

     

    def getNodeName(self):

        """This method returns the name for a given node"""

        return self.name  

   

    def getNodeLevels(self):

        """This method returns the levels or discrete values of the node"""

        return self.cpt.myVals



    def setNodeLevels(self,newVals):

        """This method can be used to set/change the levels/values of the

        node"""

        self.cpt.myVals = newVals

    

    def getNodeCPT(self):

        """This method returns the conditional probability table of the node

        if the form of a dictionary with keys as values of all combinations of

        parent nodes ."""

        '''print "The conditional probability table of " + self.name + " conditioned on parents " + str(self.parents) + " is :"'''

        return self.cpt.probTable



    def setNodeCPT(self, cpt):

        """This method is used to set the conditional probability distribution

        of a given node"""

        self.cpt = cpt

    



class DiscreteBayesNet(object): 

    """A Bayesian network with a collection of nodes"""

    def __init__(self,nodes,name):

        self.nodes=nodes

        self.variables = dict([(str(n.name), n) for n in nodes])

        self.name=name

        self.evidence = {}

    

    def getChildren(self, node):

        """This method returns the list of children nodes given a node 

        in the Bayesian Network"""

        if node not in self.nodes:

            print "The given node is not present in the network"

        else:

            children=[]

            for other in self.nodes:

                if node.name in self.getParents(other):

                    children.append(other.name)

            return children

    

    def changeNodeName(self,node,newname):

        """This method is used to change the name of a given node"""

        tempnode=node

        for i in self.nodes:

            for j in range(len(i.parents)):

                if i.parents[j]==node.name:

                    i.parents[j]=newname

        del self.variables[node.name]

        self.variables[newname]=tempnode

        node.name=newname

    

    def getMarginal(self, var):

        """This method returns the marginal distribution of a given node"""

        self.evidence={}

        marginal = self.query_ask(var,self.evidence)

        return marginal

       

    

    def getParents(self, node):

        """This method takes a node as an argument and returns the list of 

        parents for the given node. If the node is not present in the network

        tbe method throws an error message saying that the node is not present

        in the network."""

        if node not in self.nodes:

            print "The given node is not present in the network"

        else:

            structure = self.displayStructure()

            return structure[node.getNodeName()]

          

    def setEvidence(self, e):

        """This method is used to set hard evidence on the nodes. Takes the 

        evidence as argument in the form of dictionary with keys as variable

        names and values as the variable value"""

        self.evidence=e

        print "Evidence set"

    

    def getEvidence(self):

        """This method returns the evidence currently stored in the Bayesian

        network"""

        return self.evidence

        

    

    def removeEvidence(self):

        """This method is used to reset the evidence stored in the Bayesian 

        network. Resets the evidence dictionary as {}"""

        self.evidence={} 

        print "Evidence removed"

      

    def BayesInference(self,var):

        """This method takes the query variable as argument and computes the 

        posterior probability of the query variable given the hard evidence"""

        e = self.getEvidence()

        result=self.query_ask(var,e)

        return result

        

        

        

    

    def query_ask(self,var,e):        

        """Returns a dict giving value:probability mappings for a variable,

        given some known values in the network.  

        var is the string giving the target variable's name.  e is a dict giving

        variable:value mappings for known values in the network."""

        

        vals = self.variables[var].cpt.values()


        dist = {}

        if var in e:

            for v in vals:

                dist[v] = 1.0 if e[var]==v else 0.0

            return dist

            

        for v in vals:

            dist[v] = self.query_all(self.variables,extend(e, var, v))

        normalize(dist)

        return dist

      

    def query_all(self, vars, e, v=None):

        """A helper method for the query_ask method.  Gives the probability

        of the evidence in e over the variables named in vars."""


        if len(vars) == 0:

            return 1.0

            

        if v:

            Y = v

        else:

            Y = vars.keys()[0]


        Ynode = self.variables[Y]


        parents = Ynode.parents

        cpt = Ynode.cpt

        

        for p in parents:

            if p not in e: 

                return self.query_all(vars, e, p)

        

        if Y in e:

            y = e[Y]

            cp = cpt.prob_dist([e[p] for p in parents])[y]

            result = cp * self.query_all(cut(vars,Y), e)

        else:

            result = 0

            for y in Ynode.cpt.values():

                cp = cpt.prob_dist([e[p] for p in parents])[y]

                result += cp * self.query_all(cut(vars,Y),extend(e, Y, y))



        return result        

    

    def addNode(self,node):

        """This method takes a node as an argument and adds the node to the 

        Bayesian network"""

        if not isinstance(node, DiscreteBayesNode):

            print "The argument must be an object of type node"

        elif node in self.nodes:

            print "The node is already in the network"

        else:

            self.nodes.append(node)

            self.variables=extend(self.variables,node.name,node)

            



    def deleteNode(self, node):

        """This method take a node as an argument and deletes the node from

        the Bayesian network """

        if not isinstance(node, DiscreteBayesNode):

            print "The argument must be an object of type node"

        elif node not in self.nodes:

            print "Cannot delete node. The node doesn't exist"

        elif (len(self.getChildren(node)) > 0):

            print "Cannot delete parent node with child nodes"

        else:

            self.nodes.remove(node)

            del self.variables[node.name]

          

    def displayStructure(self):

        """This method displays the structure of the Bayesian network. 

        Returns a dictionary with keys as nodes and list of parents as

        values"""

        dict1={}

        for node in self.nodes: 

            parentlist=[]

            for parentname in node.parents:

                parentnode=self.variables[parentname]

                parentlist.append(str(parentnode.getNodeName()))

            dict1[str(node.name)]=parentlist

        return dict1



class NewNodeFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,150),id=wx.ID_ANY)

        nodename = wx.StaticText(self, -1,"Node Name:")

        '''values = wx.StaticText(self, -1, "Values:")'''

        parents = wx.StaticText(self, -1, "Parents:")

        self.txt_nodename = wx.TextCtrl(self, 1, size=(250,-1))

        '''self.txt_values = wx.TextCtrl(self, 1, size=(125,-1))'''

        self.txt_parents = wx.TextCtrl(self, 1, size=(250,-1))

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        sizer = wx.FlexGridSizer(rows=3, cols=2, hgap=5, vgap=10)

        sizer.Add(nodename)

        sizer.Add(self.txt_nodename)

        '''sizer.Add(values)'''

        '''sizer.Add(self.txt_values)'''

        sizer.Add(parents)

        sizer.Add(self.txt_parents)

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Create a new node')

        

                  

       

    def onSubmit(self, e):

        name = str(self.txt_nodename.GetValue())

        parents = str(self.txt_parents.GetValue()).split(',')

        if (parents==['']):

            newnode = DiscreteBayesNode(name, [])

        else:

            newnode = DiscreteBayesNode(name, parents)

        self.Close()

        dlg = wx.MessageDialog( self, "The node "+name+" has been successfully created", "About Node Creation", wx.OK)

        dlg.ShowModal()

        dlg.Destroy()

        nodenamesList.append(name)

        nodesList.append(newnode)

        

        

    

    def onReset(self, e):

        self.txt_nodename.Clear()

        self.txt_parents.Clear()

        

class NewNetworkFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,200), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        nodes = wx.StaticText(self, -1, "Nodes:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.nodes = wx.ListBox(self, choices=nodenamesList,id=1,name="Nodes ListBox",style=wx.LB_MULTIPLE)

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        self.Bind(wx.EVT_LISTBOX, self.onSelection, self.nodes)

        sizer = wx.FlexGridSizer(rows=3, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(nodes)

        sizer.Add(self.nodes)    

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Create a new network')



    def onSubmit(self,e):

        name = str(self.txt_networkname.GetValue())

        '''nodes = str(self.txt_nodes.GetValue()).split(',')'''

        nodes_selected = self.onSelection(e)

        temp = []

        for i in range(len(nodes_selected)):

            temp.append(nodesList[i])

    

        newnetwork = DiscreteBayesNet(temp, name)

        networkList.append(newnetwork)

        self.Close()

        dlg = wx.MessageDialog( self, "The network "+name+" has been successfully created", "About Network Creation", wx.OK)

        dlg.ShowModal()

        dlg.Destroy()



    def onReset(self, e):

        self.txt_networkname.Clear()



    def onSelection(self, e):

        return self.nodes.GetSelections()

        

      

class DeleteNodeFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(400,150), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        deletenode = wx.StaticText(self, -1, "Node to delete:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.deletecombo = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),nodenamesList,style=wx.CB_DROPDOWN)

        btn_Submit = wx.Button(self, -1, "&Submit")

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection, self.deletecombo)

        sizer = wx.FlexGridSizer(rows=3, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(deletenode)

        sizer.Add(self.deletecombo)    

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Delete a node from network')



    def onSubmit(self,e):

        network = str(self.txt_networkname.GetValue())

        for nw in networkList:

            if nw.name==network:

                chosen_network = nw

        node_to_delete = self.onSelection(e)

        if node_to_delete not in nodenamesList:

            dlg = wx.MessageDialog(self,"The node doesn't exist in the network","Node doesn't exist", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        else:

            nodedeleted = chosen_network.variables[node_to_delete]

            nodesList.remove(nodedeleted)

            nodenamesList.remove(node_to_delete)

            chosen_network.deleteNode(chosen_network.variables[node_to_delete])

            self.Close()

            dlg = wx.MessageDialog( self, "The node "+node_to_delete+ " has been successfully deleted from the network", "About Node Deletion", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()





    def onReset(self, e):

        self.txt_networkname.Clear()



    def onSelection(self, e):

        return str(self.deletecombo.GetValue())



class ChangeNodeNameFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,200), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        oldnodename = wx.StaticText(self, -1, "Old Node Name:")

        newnodename = wx.StaticText(self, -1, "New Node Name:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.oldnodename = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),nodenamesList,style=wx.CB_DROPDOWN)

        self.txt_newnodename = wx.TextCtrl(self, 1, size=(125,-1))

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection, self.oldnodename)

        sizer = wx.FlexGridSizer(rows=4, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(oldnodename)

        sizer.Add(self.oldnodename)

        sizer.Add(newnodename)

        sizer.Add(self.txt_newnodename)    

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Changing the name of a node in the network')



    def onSubmit(self,e):

        network = str(self.txt_networkname.GetValue())

        for nw in networkList:

            if nw.name==network:

                chosen_network = nw

        oldname = self.onSelection(e)

        newname = str(self.txt_newnodename.GetValue())

        nodesList.remove(chosen_network.variables[oldname])

        nodenamesList.remove(oldname)

        nodenamesList.append(newname)

        chosen_network.changeNodeName(chosen_network.variables[oldname],newname)

        nodesList.append(chosen_network.variables[newname])

        self.Close()

        dlg = wx.MessageDialog( self, "The node "+oldname+ " has been renamed to "+newname, "About Changing Node name", wx.OK)

        dlg.ShowModal()

        dlg.Destroy()

        

    def onReset(self, e):

        self.txt_networkname.Clear()

        self.txt_newnodename.Clear()



    def onSelection(self, e):

        return str(self.oldnodename.GetValue())

        

class DisplayNodeInfoFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,250), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        nodename = wx.StaticText(self, -1, "Node Name:")

        values = wx.StaticText(self, -1, "Node Values:")

        children = wx.StaticText(self, -1, "Node Children:")

        parents = wx.StaticText(self, -1, "Node Parents:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.nodename = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),nodenamesList,style=wx.CB_DROPDOWN)

        self.txt_values = wx.TextCtrl(self, 1, size=(250,-1))

        self.txt_children = wx.TextCtrl(self, 1, size=(250,-1))

        self.txt_parents = wx.TextCtrl(self, 1, size=(250,-1))

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection, self.nodename)

        sizer = wx.FlexGridSizer(rows=6, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(nodename)

        sizer.Add(self.nodename)

        sizer.Add(values)

        sizer.Add(self.txt_values)

        sizer.Add(children)

        sizer.Add(self.txt_children)

        sizer.Add(parents)

        sizer.Add(self.txt_parents)

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Displaying Node Information')



    def onSubmit(self,e):

        network = str(self.txt_networkname.GetValue())

        nodename = self.onSelection(e)

        temp_parents=""

        temp_children=""

        temp_values=""

        for nw in networkList:

            if nw.name==network:

                chosen_network = nw

        if nodename not in nodenamesList:

            dlg = wx.MessageDialog(self,"The node doesn't exist in the network", "Node doesn't exist", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        else:

            parents = chosen_network.getParents(chosen_network.variables[nodename])

            children = chosen_network.getChildren(chosen_network.variables[nodename])

            values = chosen_network.variables[nodename].getNodeLevels()



            for parent in parents:

                temp_parents+=parent+" "

            

            for child in children:

                temp_children+=child+" "



            for value in values:

                temp_values+=value+" "

         

            self.txt_parents.SetValue(temp_parents)

            self.txt_children.SetValue(temp_children)

            self.txt_values.SetValue(temp_values)



    def onReset(self, e):

        self.txt_parents.Clear()

        self.txt_children.Clear()

        self.txt_values.Clear()



    def onSelection(self,e):

        return str(self.nodename.GetValue())



        

class SetEvidenceFrame(wx.Frame):

    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,200), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        nodename = wx.StaticText(self, -1, "Node Name:")

        setnodevalue = wx.StaticText(self, -1, "Set Node Value:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.nodename = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),nodenamesList,style=wx.CB_DROPDOWN)

        self.setnodevalue = wx.ComboBox(self,500,"No value selected",(50, 150), (125,-1),[],style=wx.CB_DROPDOWN)

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_AddNewEvidence = wx.Button(self, -1, "&Add New Evidence")

        self.Bind(wx.EVT_BUTTON, self.onAddNewEvidence, btn_AddNewEvidence)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection1, self.nodename)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection2, self.setnodevalue)

        sizer = wx.FlexGridSizer(rows=4, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(nodename)

        sizer.Add(self.nodename)

        sizer.Add(setnodevalue)

        sizer.Add(self.setnodevalue)    

        sizer.Add(btn_Submit)

        sizer.Add(btn_AddNewEvidence)

        self.SetSizer(sizer)

        self.SetTitle('Setting hard evidence')



    def onSubmit(self,e):

        networkname = str(self.txt_networkname.GetValue())

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        nodename = self.onSelection1(e)

        nodevalue = self.onSelection2(e)

        custom_network.evidence[nodename]=nodevalue

        for key in custom_network.evidence.keys():

            if key not in evidenceList:

                evidenceList.append(key)

        self.Close()

        dlg = wx.MessageDialog( self, "Evidence added successfully", "Setting Evidence", wx.OK)

        dlg.ShowModal()

        dlg.Destroy()



    def onAddNewEvidence(self, e):

        networkname = str(self.txt_networkname.GetValue())

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        nodename = str(self.nodename.GetValue())

        nodevalue = str(self.setnodevalue.GetValue())

        custom_network.evidence[nodename]=nodevalue

        for key in custom_network.evidence.keys():

            if key not in evidenceList:

                evidenceList.append(key)

        self.nodename.SetValue('No node selected')

        self.setnodevalue.SetValue('No value selected')



    def onSelection1(self,e):

        nodename = str(self.nodename.GetValue())

        return nodename



    def onSelection2(self,e):

        network = str(self.txt_networkname.GetValue())

        nodename = str(self.nodename.GetValue())

        for nw in networkList:

            if nw.name==network:

                chosen_network = nw

        nodeValues = chosen_network.variables[nodename].getNodeLevels()

        self.setnodevalue.AppendItems(nodeValues)

        return str(self.setnodevalue.GetValue())

        



class RemoveEvidenceFrame(wx.Frame):

    def __init__(self, parent):

        wx.Frame.__init__(self, parent,size=(450,200), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        nodename = wx.StaticText(self, -1, "Node Name:")

        getnodeevidence = wx.StaticText(self, -1, "Node Evidence:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.nodename = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),evidenceList,style=wx.CB_DROPDOWN)

        self.txt_nodeevidence = wx.TextCtrl(self, 1, size=(125,-1))

        btn_getNodeEvidence = wx.Button(self, -1, "&Get Node Evidence")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_getNodeEvidence)

        btn_RemoveEvidence = wx.Button(self, -1, "&Remove Node Evidence")

        self.Bind(wx.EVT_BUTTON, self.onRemoveEvidence, btn_RemoveEvidence)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection, self.nodename)

        sizer = wx.FlexGridSizer(rows=4, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(nodename)

        sizer.Add(self.nodename)

        sizer.Add(getnodeevidence)

        sizer.Add(self.txt_nodeevidence)    

        sizer.Add(btn_getNodeEvidence)

        sizer.Add(btn_RemoveEvidence)

        self.SetSizer(sizer)

        self.SetTitle('Removing hard evidence')



    def onRemoveEvidence(self,e):

        networkname = str(self.txt_networkname.GetValue())

        nodename = str(self.nodename.GetValue())

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        if nodename not in nodenamesList:

            dlg = wx.MessageDialog(self,"The node doesn't exist in the network", "Node doesn't exist", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        elif nodename not in custom_network.evidence.keys():

            dlg = wx.MessageDialog(self,"The node has not been used as hard evidence ", "Node not used as hard evidence", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        else:

            del custom_network.evidence[nodename]

            evidenceList.remove(nodename)

        self.Close()

        dlg = wx.MessageDialog( self, "Evidence removed successfully", "Removing Evidence", wx.OK)

        dlg.ShowModal()

        dlg.Destroy()



    def onSubmit(self, e):

        networkname = str(self.txt_networkname.GetValue())

        nodename = self.onSelection(e)

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        if nodename not in nodenamesList:

            dlg = wx.MessageDialog(self,"The node doesn't exist in the network", "Node doesn't exist", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        elif nodename not in custom_network.evidence.keys():

            dlg = wx.MessageDialog(self,"The node has not been used as hard evidence ", "Node not used as hard evidence", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        else:

            self.txt_nodeevidence.SetValue(custom_network.evidence[nodename])

            

        

    def onSelection(self, e):

        return str(self.nodename.GetValue())

    



class GetMarginalFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,250), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        nodename = wx.StaticText(self, -1, "Node Name:")

        values = wx.StaticText(self, -1, "Node Values:")

        marginal = wx.StaticText(self, -1, "Marginal Probabilities:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.nodename = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),nodenamesList,style=wx.CB_DROPDOWN)

        self.txt_values = wx.TextCtrl(self, 1, size=(250,-1))

        self.txt_prob = wx.TextCtrl(self, 1, size=(250,-1))

        btn_Submit = wx.Button(self, -1, "&Get Marginal")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection, self.nodename)

        sizer = wx.FlexGridSizer(rows=5, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(nodename)

        sizer.Add(self.nodename)

        sizer.Add(values)

        sizer.Add(self.txt_values)

        sizer.Add(marginal)

        sizer.Add(self.txt_prob)

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Get Marginal Probabilities')



    def onSubmit(self,e):

        networkname = str(self.txt_networkname.GetValue())

        nodename = self.onSelection(e)

        temp_values=""

        temp_probs=""

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        if nodename not in nodenamesList:

            dlg = wx.MessageDialog(self,"The node doesn't exist in the network", "Node doesn't exist", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        else:

            values = custom_network.variables[nodename].getNodeLevels()

            for value in values:

                temp_values+=value+" "

    

            self.txt_values.SetValue(temp_values)

            marginals = custom_network.getMarginal(nodename)

          

            for value in values:

                temp_probs+=str(round(marginals[value],4))+" "

    

            self.txt_prob.SetValue(temp_probs)

            



    def onReset(self, e):

        self.nodename.SetValue('No node selected')

        self.txt_values.Clear()

        self.txt_prob.Clear()



    def onSelection(self,e):

        return str(self.nodename.GetValue())



class BayesInferenceFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent,size=(450,250), id=wx.ID_ANY)

        networkname = wx.StaticText(self, -1, "Network Name:")

        nodename = wx.StaticText(self, -1, "Node Name:")

        values = wx.StaticText(self, -1, "Node Values:")

        conditional = wx.StaticText(self, -1, "Conditional Probabilities:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.nodename = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),nodenamesList,style=wx.CB_DROPDOWN)

        self.txt_values = wx.TextCtrl(self, 1, size=(250,-1))

        self.txt_prob = wx.TextCtrl(self, 1, size=(250,-1))

        btn_Submit = wx.Button(self, -1, "&Get Conditional Probabilities")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection, self.nodename)

        sizer = wx.FlexGridSizer(rows=5, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)                                

        sizer.Add(nodename)

        sizer.Add(self.nodename)

        sizer.Add(values)

        sizer.Add(self.txt_values)

        sizer.Add(conditional)

        sizer.Add(self.txt_prob)

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Get Conditional Probabilities')



    def onSubmit(self,e):

        networkname = str(self.txt_networkname.GetValue())

        nodename = self.onSelection(e)

        temp_values=""

        temp_probs=""

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        if nodename not in nodenamesList:

            dlg = wx.MessageDialog(self,"The node doesn't exist in the network", "Node doesn't exist", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()

        else:

            values = custom_network.variables[nodename].getNodeLevels()

            for value in values:

                temp_values+=value+" "



            self.txt_values.SetValue(temp_values)



            conditionals = custom_network.BayesInference(nodename)

            for value in values:

                temp_probs+=str(round(conditionals[value],4))+" "

        

            self.txt_prob.SetValue(temp_probs)

                

    def onReset(self, e):

        self.nodename.SetValue('No node selected')

        self.txt_values.Clear()

        self.txt_prob.Clear()



    def onSelection(self,e):

        return str(self.nodename.GetValue())



class SetNodeCPTFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,300),id=wx.ID_ANY)

        nodename = wx.StaticText(self, -1, "Node Name:")

        values = wx.StaticText(self, -1, "Node Values:")

        parentvalues = wx.StaticText(self, -1, "Parent Values:")

        conditional = wx.StaticText(self, -1, "Conditional Probabilities:")

        self.nodename = wx.ComboBox(self,500,"No node selected",(50, 150), (125,-1),nodenamesList,style=wx.CB_DROPDOWN)

        self.txt_values = wx.TextCtrl(self, 1, size=(250,-1))

        self.txt_parentvalues = wx.TextCtrl(self, 1, size=(250,-1),style=wx.TE_MULTILINE)

        self.txt_probs = wx.TextCtrl(self, 1, size=(250,-1),style=wx.TE_MULTILINE)

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Reset = wx.Button(self, -1, "&Reset")

        self.Bind(wx.EVT_BUTTON, self.onReset, btn_Reset)

        self.Bind(wx.EVT_COMBOBOX, self.onSelection, self.nodename)

        sizer = wx.FlexGridSizer(rows=5, cols=2, hgap=5, vgap=10)

        sizer.Add(nodename)

        sizer.Add(self.nodename)

        sizer.Add(values)

        sizer.Add(self.txt_values)

        sizer.Add(parentvalues)

        sizer.Add(self.txt_parentvalues)

        sizer.Add(conditional)

        sizer.Add(self.txt_probs)

        sizer.Add(btn_Submit)

        sizer.Add(btn_Reset)

        self.SetSizer(sizer)

        self.SetTitle('Set Conditional Probabilities')



    def onSubmit(self,e):

        probTable2={}

        parentcombs=[]

        probscomb=[]

        nodename = self.onSelection(e)

        values = str(self.txt_values.GetValue()).split(',')

        parentvaluepairs = str(self.txt_parentvalues.GetValue()).split(';')



        if parentvaluepairs == ['']:

            probs = str(self.txt_probs.GetValue()).split(',')

            probTable = map(float, probs)

            cpt = DiscreteCPT(values, probTable)     

        else:

            probs = str(self.txt_probs.GetValue()).split(';')

            for p in parentvaluepairs:

                parentcombs.append(p.split(','))

            parentcombs = map(tuple, parentcombs)

            for x in probs:

                temp = x.split(',')

                temp = map(float, temp)

                probscomb.append(temp)

            for i in range(len(parentcombs)):

                probTable2[parentcombs[i]] = probscomb[i]

            cpt = DiscreteCPT(values, probTable2)

        for node in nodesList:

            if node.name==nodename:

                node.setNodeCPT(cpt)

        self.Close()

        dlg = wx.MessageDialog( self, "Successfully set CPT for node "+nodename, "Setting CPT", wx.OK)

        dlg.ShowModal()

        dlg.Destroy()



    def onReset(self, e):

        self.txt_values.Clear()

        self.txt_parentvalues.Clear()

        self.txt_probs.Clear()



    def onSelection(self,e):

        return str(self.nodename.GetValue())



class SaveJsonFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent,size=(350,150), id=wx.ID_ANY)

        self.filename = ""

        self.dirname = ""

        networkname = wx.StaticText(self, -1, "Network Name:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Close = wx.Button(self, -1, "&Close")

        self.Bind(wx.EVT_BUTTON, self.onExit, btn_Close)

        sizer = wx.FlexGridSizer(rows=2, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)   

        sizer.Add(btn_Submit)

        sizer.Add(btn_Close)

        self.SetSizer(sizer)

        self.SetTitle('Save network as JSON')



    def onSubmit(self, e):

        networkname = str(self.txt_networkname.GetValue())

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        dlg = wx.FileDialog(self, "Choose a file", defaultDir=self.dirname, 

                            defaultFile=self.filename, style=wx.FD_SAVE)

        if dlg.ShowModal() == wx.ID_OK:

            self.filename = dlg.GetFilename()

            self.dirname = dlg.GetDirectory()

            fullname = os.path.join(self.dirname, self.filename)

            json_serialize(custom_network,fullname)

            dlg = wx.MessageDialog( self, "The network "+networkname+ " has been successfully saved in JSON", "About Save to JSON", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()



            

        dlg.Destroy()

        self.Close()



    def onExit(self,e):

        self.Close()



  

class DisplayNetworkStructureFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent,size=(450,150), id=wx.ID_ANY)

        self.filename = ""

        self.dirname = ""

        networkname = wx.StaticText(self, -1, "Network Name:")

        structure = wx.StaticText(self, -1, "Network Structure:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        self.txt_structure = wx.TextCtrl(self, 1, size=(250,-1),style=wx.TE_MULTILINE)

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Close = wx.Button(self, -1, "&Close")

        self.Bind(wx.EVT_BUTTON, self.onExit, btn_Close)

        sizer = wx.FlexGridSizer(rows=3, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(structure)

        sizer.Add(self.txt_structure)

        sizer.Add(btn_Submit)

        sizer.Add(btn_Close)

        self.SetSizer(sizer)

        self.SetTitle('Display Network Structure')



    def onSubmit(self, e):

        networkname = str(self.txt_networkname.GetValue())

        structure={}

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        structure = custom_network.displayStructure()

        self.txt_structure.SetValue(str(structure))

    



    def onExit(self,e):

        self.Close()                                           

                                        

class DisplayNetworkGraphFrame(wx.Frame):



    def __init__(self, parent):

        wx.Frame.__init__(self, parent, size=(450,150), id=wx.ID_ANY)

        self.filename = ""

        self.dirname = ""

        networkname = wx.StaticText(self, -1, "Network Name:")

        self.txt_networkname = wx.TextCtrl(self, 1, size=(125,-1))

        btn_Submit = wx.Button(self, -1, "&Submit")

        self.Bind(wx.EVT_BUTTON, self.onSubmit, btn_Submit)

        btn_Close = wx.Button(self, -1, "&Close")

        self.Bind(wx.EVT_BUTTON, self.onExit, btn_Close)

        sizer = wx.FlexGridSizer(rows=3, cols=2, hgap=5, vgap=10)

        sizer.Add(networkname)

        sizer.Add(self.txt_networkname)

        sizer.Add(btn_Submit)

        sizer.Add(btn_Close)

        self.SetSizer(sizer)

        self.SetTitle('Display Network Graph')



    def onSubmit(self, e):

        networkname = str(self.txt_networkname.GetValue())

        for nw in networkList:

            if nw.name==networkname:

                custom_network=nw

        fig = display(custom_network)

        fig.show()

    



    def onExit(self,e):

        self.Close() 



class NewHelpFrame(wx.Frame):



    def __init__(self,parent):

        wx.Frame.__init__(self, parent,size=(700,600), id=wx.ID_ANY)

        self.filename = ""

        self.dirname = ""

        self.panel = wx.Panel(self)    

        self.panel.SetBackgroundColour('white')

        

        tit1 = "Functions"

        

        sub1 = "File"

        str1 = "1. New Node - Create a new node"

        str2 = "2. New Bayesian Network - Create a new Bayesian Network"

        str3 = "3. Load Bayesian Network from JSON - Load a Bayesian Network from JSON file"

        str4 = "4. Load Bayesian Network from JSON - Load a Bayesian Network from JSON file"

        sub2 = "Edit"        

        str5 = "1. Delete Node - Delete an existing node"

        str6 = "2. Change Node Name - Change the name of an existing node"

        str7 = "3. Set Conditional Probability Table - Set the conditional probability of an existing node"

        sub3 = "Display"

        str8 = "1. Display Node Information - Display basic information of an existing node"

        str9 = "2. Display Bayesian Network Structure - Display basic constitution of an existing Bayesian Network"

        str10 = "3. Display Bayesian Network Graph - Display the visual representation of a Bayesian Network"

        sub4 = "Inference"

        str11 = "1. Set Evidence - Set hard evidence on an existing node"

        str12 = "2. Remove Evidence - Remove hard evidence from an existing node"

        str13 = "3. Get Marginal Distribution - Get marginal distribution of an existing node"

        str14 = "4. Bayesian Inference - Do Bayesian Inference (get belief)"

        sub5= "Help"

        str15 = "1. About SimpleBayesNet - Get some knowledge about SimpleBayesNet"

        str16 = "2. SimpleBayesNet Help - A help documentation"

        

        title1 = wx.StaticText(self.panel, -1, tit1,(10,10))

        help1 = wx.StaticText(self.panel, -1, sub1,(10,50))

        help2 = wx.StaticText(self.panel, -1, str1,(30,70))

        help3 = wx.StaticText(self.panel, -1, str2,(30,90)) 

        help4 = wx.StaticText(self.panel, -1, str3,(30,110))

        help5 = wx.StaticText(self.panel, -1, str4,(30,130))

        help6 = wx.StaticText(self.panel, -1, sub2,(10,150))    

        help7 = wx.StaticText(self.panel, -1, str5,(30,170)) 

        help8 = wx.StaticText(self.panel, -1, str6,(30,190)) 

        help9 = wx.StaticText(self.panel, -1, str7,(30,210)) 

        help10 = wx.StaticText(self.panel, -1, sub3,(10,230)) 

        help11 = wx.StaticText(self.panel, -1, str8,(30,250)) 

        help12 = wx.StaticText(self.panel, -1, str9,(30,270)) 

        help13 = wx.StaticText(self.panel, -1, str10,(30,290)) 

        help14 = wx.StaticText(self.panel, -1, sub4,(10,310))

        help15 = wx.StaticText(self.panel, -1, str11,(30,330))

        help16 = wx.StaticText(self.panel, -1, str12,(30,350))

        help17 = wx.StaticText(self.panel, -1, str13,(30,370))                        

        help18 = wx.StaticText(self.panel, -1, str14,(30,390))    

        help19 = wx.StaticText(self.panel, -1, sub5,(10,410))

        help20 = wx.StaticText(self.panel, -1, str15,(30,430))  

        help21 = wx.StaticText(self.panel, -1, str16,(30,450))    

        

        fontTitle = wx.Font(15, wx.DECORATIVE, wx.NORMAL, wx.BOLD)

        fontText = wx.Font(10, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)

        fontHighlight = wx.Font(10, wx.DECORATIVE, wx.NORMAL, wx.BOLD)

        

        title1.SetFont(fontTitle)

        help1.SetFont(fontHighlight)

        help6.SetFont(fontHighlight)

        help10.SetFont(fontHighlight)

        help14.SetFont(fontHighlight)

        help19.SetFont(fontHighlight)        

        help2.SetFont(fontText)

        help3.SetFont(fontText)

        help4.SetFont(fontText)

        help5.SetFont(fontText)

        help7.SetFont(fontText)

        help8.SetFont(fontText)

        help9.SetFont(fontText)

        help11.SetFont(fontText)

        help12.SetFont(fontText)

        help13.SetFont(fontText)

        help15.SetFont(fontText)

        help16.SetFont(fontText) 

        help17.SetFont(fontText)

        help18.SetFont(fontText) 

        help20.SetFont(fontText)

        help21.SetFont(fontText) 

              

        self.SetTitle('Help')



        

        

class MainWindow(wx.Frame):

    

    def __init__(self, *args, **kwargs):

        super(MainWindow, self).__init__(*args, **kwargs) 

        self.panel = wx.Panel(self)    

        self.panel.SetBackgroundColour('white')

        self.InitUI()

        

        str1 = 'Welcome to SimpleBayesNet!'

        str2 = 'This is a Bayesian Network Framework for creating and querying a Bayesian Network developed by Ling and Anirban.'

        str3 = 'If it is your first time to use SimpleBayesNet, please read the step-by-step guide or click Help in Menu bar to learn more.'

        str4 = 'Otherwise, please click File in Menu bar to start building your own Bayesian Network.'

        str5 = 'Thank you for choosing SimpleBayesNet! Enjoy!'

        

        title = wx.StaticText(self.panel, -1,str1,(100,10),style=wx.ALIGN_CENTER)

        text1 = wx.StaticText(self.panel, -1, str2,(100,45),style=wx.ALIGN_LEFT)

        text2 = wx.StaticText(self.panel, -1, str3,(100,70),style=wx.ALIGN_LEFT)

        text3 = wx.StaticText(self.panel, -1, str4,(100,90),style=wx.ALIGN_LEFT)

        text4 = wx.StaticText(self.panel, -1, str5,(100,110),style=wx.ALIGN_CENTER)

        

        fontTitle = wx.Font(18, wx.DECORATIVE, wx.NORMAL, wx.BOLD)

        fontText = wx.Font(10, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)

        fontHighlight = wx.Font(10, wx.DECORATIVE, wx.NORMAL, wx.BOLD)

        

        title.SetFont(fontTitle)

        text1.SetFont(fontText)

        text2.SetFont(fontText)

        text3.SetFont(fontText)

        text4.SetFont(fontHighlight)

        

        self.filename=""

        self.dirname=""

        

    def InitUI(self):

        menubar = wx.MenuBar()

        fileMenu = wx.Menu()

        menuNewNode = fileMenu.Append(wx.ID_NEW, '&New Node',"Create a new node")

        menuNewNetwork = fileMenu.Append(wx.ID_ANY, '&New Bayesian Network',"Create a new Bayesian Network")

        menuLoadJson = fileMenu.Append(wx.ID_OPEN, '&Load Bayesian Network from JSON',"Load a Bayesian Network from JSON file")

        menuSaveJson = fileMenu.Append(wx.ID_SAVE, '&Save Bayesian Network to JSON',"Save a Bayesian Network to JSON file")

        fileMenu.AppendSeparator()

        '''menuAbout = fileMenu.Append(wx.ID_ABOUT, "&About","Information about this program")'''

        qmi = wx.MenuItem(fileMenu, wx.ID_EXIT, '&Quit\tCtrl+Q')

        fileMenu.AppendItem(qmi)



        editMenu = wx.Menu()

        menuDeleteNode = editMenu.Append(wx.ID_ANY, '&Delete Node',"Delete an existing node")

        menuChangeNodeName = editMenu.Append(wx.ID_ANY, '&Change Node Name',"Change the name of an existing node")

        menuSetCPT = editMenu.Append(wx.ID_ANY, '&Set Conditional Probability Table',"Set the conditional probability of an existing node")



        displayMenu = wx.Menu()

        menuNodeInfo = displayMenu.Append(wx.ID_ANY, '&Display Node Information',"Display basic information of an existing node")

        '''displayMenu.Append(wx.ID_ANY, '&Display Parents')'''

        menudisplayStructure = displayMenu.Append(wx.ID_ANY, '&Display Bayesian Network Structure',"Display basic constitution of an existing Bayesian Network")

        menudisplayGraph = displayMenu.Append(wx.ID_ANY, '&Display Bayesian Network Graph',"Display the visual representation of a Bayesian Network")



        inferenceMenu = wx.Menu()

        menuSetEvidence = inferenceMenu.Append(wx.ID_ANY, '&Set Evidence',"Set hard evidence on an existing node")

        menuRemoveEvidence = inferenceMenu.Append(wx.ID_ANY, '&Remove Evidence',"Remove hard evidence from an existing node")

        menuGetMarginal = inferenceMenu.Append(wx.ID_ANY, '&Get Marginal Distribution',"Get marginal distribution of an existing node")

        menuBayesInference = inferenceMenu.Append(wx.ID_ANY, '&Bayesian Inference',"Do Bayesian Inference - get belief")



        helpMenu = wx.Menu()

        menuAbout = helpMenu.Append(wx.ID_ANY, '&About SimpleBayesNet',"About SimpleBayesNet")

        helpMenu.AppendSeparator()

        menuHelp = helpMenu.Append(wx.ID_ANY, '&SimpleBayesNet Help',"Help Documentation")

        



        

        self.Bind(wx.EVT_MENU, self.OnQuit, qmi)

        self.Bind(wx.EVT_MENU, self.OnAboutBox, menuAbout)

        self.Bind(wx.EVT_MENU, self.OnHelpBox, menuHelp)

        self.Bind(wx.EVT_MENU, self.OnNewNode, menuNewNode)

        self.Bind(wx.EVT_MENU, self.OnNewNetwork, menuNewNetwork)

        self.Bind(wx.EVT_MENU, self.OnLoadJson, menuLoadJson)

        self.Bind(wx.EVT_MENU, self.OnSaveJson, menuSaveJson)

        self.Bind(wx.EVT_MENU, self.OnDeleteNode, menuDeleteNode)

        self.Bind(wx.EVT_MENU, self.OnChangeNodeName, menuChangeNodeName)

        self.Bind(wx.EVT_MENU, self.OnDisplayNodeInfo, menuNodeInfo)

        self.Bind(wx.EVT_MENU, self.OnSetEvidence, menuSetEvidence)

        self.Bind(wx.EVT_MENU, self.OnRemoveEvidence, menuRemoveEvidence)

        self.Bind(wx.EVT_MENU, self.OnGetMarginal, menuGetMarginal)

        self.Bind(wx.EVT_MENU, self.OnBayesInference, menuBayesInference)

        self.Bind(wx.EVT_MENU, self.OnSetCPT, menuSetCPT)

        self.Bind(wx.EVT_MENU, self.OnDisplayStructure, menudisplayStructure)

        self.Bind(wx.EVT_MENU, self.OnDisplayGraph, menudisplayGraph)

        

        menubar.Append(fileMenu, '&File')

        menubar.Append(editMenu, '&Edit')

        menubar.Append(displayMenu, '&Display')

        menubar.Append(inferenceMenu, '&Inference')

        menubar.Append(helpMenu, '&Help')

        self.SetMenuBar(menubar)



        self.SetSize((800, 400))

        self.CreateStatusBar(2) 

        self.SetStatusWidths([-1, 200])



        today = datetime.datetime.today()

        today = today.strftime('%d-%b-%Y')



        self.SetStatusText(today, 1)

        self.SetTitle('SimpleBayesNet')

        self.Centre()

        self.Show(True)

        

    def OnQuit(self, e):

        self.Close()



    def OnAboutBox(self,e):

        description = """SimpleBayesNet is an advanced Bayesian Network framework. Features include creating a Bayesian network from scratch, displaying the Bayesian network generated and querying the

Bayesian network for Bayesian inference

"""



        info = wx.AboutDialogInfo()

        info.SetName('SimpleBayesNet')

        info.SetVersion('1.0')

        info.SetDescription(description)

        info.AddDeveloper('Ling Jin and Anirban Bhattacharyya')

        wx.AboutBox(info)

        '''                            

        dlg = wx.MessageDialog( self, "A Bayesian Network GUI developed by Ling and Anirban", "About SimpleBayesNet", wx.OK)

        dlg.ShowModal()

        dlg.Destroy()

        '''

    def OnHelpBox(self,e):

        helpframe = NewHelpFrame(None)

        helpframe.Show()

        

    def OnNewNode(self,e):

        nodeframe = NewNodeFrame(None)

        nodeframe.Show()



    def OnNewNetwork(self,e):

        networkframe = NewNetworkFrame(None)

        networkframe.Show()



    def OnLoadJson(self,e):

        dlg = wx.FileDialog(self, "Choose a file", defaultDir=self.dirname, 

                            defaultFile=self.filename, style=wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:

            self.filename = dlg.GetFilename()

            self.dirname = dlg.GetDirectory()

            fullname = os.path.join(self.dirname, self.filename)

            bn = json_load_file(fullname)

            networkList.append(bn)

        

            for n in bn.variables.keys():

                del bn.variables[n]

          

            for node in bn.nodes:

                bn.variables[str(node.name)]=node


            for node in nodesList:

                nodesList.remove(node)

                nodenamesList.remove(str(node.name))


            for node in bn.nodes:

                nodesList.append(node)

                nodenamesList.append(str(node.name))

          
            dlg = wx.MessageDialog(self, "The network has been successfully loaded from the JSON", "About Load JSON", wx.OK)

            dlg.ShowModal()

            dlg.Destroy()



            # Set filename with path in the first field of the status bar

            self.DisplayFilenameOnStatusBar()

            '''

            f = open(fullname, 'r')        

            self.control.SetValue(f.read())

            f.close()

            '''

        dlg.Destroy()



    def OnSaveJson(self, e):

        frame = SaveJsonFrame(None)

        frame.Show()

        
    def OnDeleteNode(self,e):

        frame = DeleteNodeFrame(None)

        frame.Show()


    def OnChangeNodeName(self,e):

        frame = ChangeNodeNameFrame(None)

        frame.Show()

    


    def OnDisplayNodeInfo(self,e):

        frame = DisplayNodeInfoFrame(None)

        frame.Show()



    def OnSetEvidence(self,e):

        frame = SetEvidenceFrame(None)

        frame.Show()



    def OnRemoveEvidence(self, e):

        frame = RemoveEvidenceFrame(None)

        frame.Show()



    def OnGetMarginal(self, e):

        frame = GetMarginalFrame(None)

        frame.Show()



    def OnBayesInference(self, e):

        frame = BayesInferenceFrame(None)

        frame.Show()



    def OnSetCPT(self, e):

        frame = SetNodeCPTFrame(None)

        frame.Show()



    def OnDisplayStructure(self, e):

        frame = DisplayNetworkStructureFrame(None)

        frame.Show()

    

    def OnDisplayGraph(self, e):

        frame = DisplayNetworkGraphFrame(None)

        frame.Show()



    def DisplayFilenameOnStatusBar(self):

        'Display the filename in the Status bar'

        fullname = os.path.join(self.dirname, self.filename)

        # Set filename with path in the first field of the status bar

        self.SetStatusText(fullname, 0)

                                  



def main():

    

    app = wx.App()

    frame=MainWindow(None)

    app.MainLoop()    





if __name__ == '__main__':

    main()

