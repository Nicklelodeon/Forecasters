import numpy as np
from Item import Item
from Stochastic_Lead_Time import Stochastic_Lead_Time
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 

class Demandable:
    def __init__(self, name, holding_cost, backorder_cost, s, S):
        self.name = name
        self.inv_level = {}  ## Each item has multiple inv level
        self.inv_pos = {}
        self.int_map_to_inv = []
        self.inv_map = {} ## Has the inventory to Demandable
        self.upstream = []  ## Each upstream Demandables
        self.downstream = [] ## Each downstream Demandables
        self.holding_cost = holding_cost 
        self.ordering_costs = []
        self.holding_costs = []
        self.backorder_costs = []
        self.inv_level_plot = []

        self.backorder = 0
        self.backorder_cost = backorder_cost
        self.stochastic_lead_time = None
        self.lead_time = [-1, 0] #Time and lead_time
        
        self.costs = []
        self.arrivals = []
        self.total_costs = 0
        self.orders = {}

    def reset(self, amount=65):
        self.inv_level  = dict.fromkeys(self.inv_level, amount)
        self.inv_pos = dict.fromkeys(self.inv_pos, amount)
        self.ordering_costs = []
        self.holding_costs = []
        self.backorder_costs = []
        self.backorder = 0
        self.costs = []
        self.arrivals = []
        self.total_costs = 0
        self.inv_level_plot = []
    
    def get_state(self):
        lst = np.zeros(2 * len(self.int_map_to_inv))
        
        for i in range(len(self.int_map_to_inv)):
            item = self.int_map_to_inv[i]
            item_pos = self.inv_pos[item]
            item_level = self.inv_level[item]
            lst[i*2] = item_pos
            lst[i*2+1] = item_level
            
        return lst

    def add_lead_time(self, stl):
        """Assign stochastic lead time

        Args:
            stl (Stochastic Lead Time): Samples lead time from distribution
        """
        self.stochastic_lead_time = stl

    def update_demand(self, num_get: int):
        """Update inv level and inv pos and cost

        Args:
            num_get (int): amount requested 

        Returns:
            items_out (int): amount available to be ordered
        """

        if self.inv_level:
            min_item = min(min(list(self.inv_level.values())), num_get)
        else:
            min_item = num_get
            
        curr_backorder = num_get - min_item
        self.backorder += curr_backorder
        for item in self.inv_level:
            self.inv_level[item] -= min_item
            self.inv_pos[item] -= num_get
        return min_item
    
    def get_lead_time(self, t):
        """Gets lead time for time t

        Args:
            t (int): time stamp

        Returns:
            int: lead time
        """
        if t == self.lead_time[0]:
            return self.lead_time[1]
        else:
            new_lead_time = self.stochastic_lead_time.get_lead_time()
            self.lead_time = [t, new_lead_time]
            return new_lead_time

    """ def check_s(self, item, t):
        recursively checks if inv pos < s for all upstream demandables. Adds order cost to total cost

        Args:
            item (Item): item to be ordered
            t (int): time stamp
        
        if self.inv_pos[item] < self.s:
            if item in self.inv_map:
                demandable = self.inv_map[item]
                amt = self.S - self.inv_pos[item]
                ordered_amt = demandable.produce_order(item, amt)
                if ordered_amt > 0:
                    lead_time = demandable.get_lead_time(t)
                    #lead_time = self.stochastic_lead_time.get_lead_time() #removed to get constant lead time for each dc at each t
                    self.arrivals.append([t + lead_time, item, ordered_amt])
                    self.ordering_costs[t] += ordered_amt * item.get_cost()
                    self.total_costs += ordered_amt * item.get_cost()
                demandable.check_s(item, t) """
    
    def order_item(self, amt, t):
        #pos: [Item(1): 20, Item(2): 15], amt = 20 -> [Item(1): 40, Item(2): 40]
        max_position = max(self.inv_pos.values())
        top_up_to = max_position + amt 
        
        for item in self.int_map_to_inv:
            demandable = self.inv_map[item]
            curr_item_level = self.inv_pos[item]
            attempt_to_order = top_up_to - curr_item_level
            ordered_amt = demandable.produce_order(item, attempt_to_order)
            
            if ordered_amt > 0:
                lead_time = demandable.get_lead_time(t)
                self.arrivals.append([t + lead_time, item, ordered_amt])
                self.ordering_costs[t] += ordered_amt * item.get_cost()
                self.total_costs += ordered_amt * item.get_cost()
            
                self.inv_pos[item] += ordered_amt
            
    
    """ def order_item(self, integer, amt, t):
        item = self.int_map_to_inv[integer]
        demandable = self.inv_map[item]
        ordered_amt = demandable.produce_order(item, amt)
        
        if ordered_amt > 0:
            lead_time = demandable.get_lead_time(t)
            self.arrivals.append([t + lead_time, item, ordered_amt])
            self.ordering_costs[t] += ordered_amt * item.get_cost()
            self.total_costs += ordered_amt * item.get_cost()
            
            self.inv_pos[item] += ordered_amt """

    def produce_order(self, item, amt):
        """Determine amount to be ordered

        Args:
            item (Item): item to be ordered
            amt (int): amount requested

        Returns:
            int: amount available to be ordered
        """
        amt_avail = self.inv_level[item]
        amt_supplied = min(amt_avail, amt)
        self.inv_level[item] -= amt_supplied
        self.inv_pos[item] -= amt_supplied
        return amt_supplied

    def add_upstream(self, demandable: "Demandable") -> None:
        """Adds a demandable into upstream

        Args:
            demandable (Demandable): To be added in upstream
        """
        self.upstream.append(demandable)
        demandable.add_downstream(self)
        # Change later, perhaps random starting inventory ISSUE here

    def add_downstream(self, demandable: "Demandable") -> None:
        """Adds a demandable into downstream, called after
        add_upstream function

        Args:
            demandable (Demandable): demandable
        """
        self.downstream.append(demandable)
    
    def find_changeable_network(self):
        """Finds retailers and distribution centres in the network

        Returns:
            List<Demandable>: list of retailers and distribution centres
        """
        list = []
        if self.upstream:
            list += [self]
            for demandable in self.upstream:
                list += demandable.find_changeable_network()
        return list
            
    def add_item_map(self, item, demandable):
        """Maps item to Demandable in self.inv_map

        Args:
            item (Item): An Item
            Demandable (Demandable): Direct upstream Demandable
        """
        self.int_map_to_inv.append(item)
        self.inv_map[item] = demandable
        
    def add_item_downstream(self, item, amount=65):
        """Adds items to all the downstream

        Args:
            item (Item): Item added
            amount (int): amount of item to be adde
        """
        self.add_item(item, amount)
        if self.downstream: # Check if list empty
            downstream_demandable = self.downstream[0]
            downstream_demandable.add_item_map(item, self)
            downstream_demandable.add_item_downstream(item, amount)    

    def add_item(self, item: "Item", amt = 0):
        """Add item to demandable and its downstream, create inv level and inv pos

        Args:
            item (Item): Item added
            amt (int): Amount of item to be added
        """
        self.inv_level[item] = amt
        self.inv_pos[item] = amt

    def find_end_upstream(self) -> list:
        """Finds the topmost upstream demandable

        Returns:
            list of demandables 
        """
        leaves = []
        if self.upstream: #upstream not empty
            for demandable in self.upstream:
                leaves += demandable.find_end_upstream()
        else:
            leaves += [self]
        return leaves
            
    def update_inventory(self, t):
        """Updates inv level and inv pos

        Args:
            t (int): timestamp
        """
        # self.inv_pos = self.inv_level.copy()
        index = []
        for i in range(len(self.arrivals)):
            arrival = self.arrivals[i]
            time, item, amt = arrival
            if t == time:
                self.inv_level[item] += amt
                index.append(i)
            # self.inv_pos[item] += amt
        self.arrivals = [arrival for i, arrival in enumerate(self.arrivals) if i not in index]
        if self.backorder > 0:
            amt_backordered = min(self.backorder, min(list(self.inv_level.values())))
            for item in self.inv_level:
                self.inv_level[item] -= amt_backordered
            self.backorder -= amt_backordered

    def update_all_inventory(self, t):
        """Updates inv level for all upstream demandables

        Args:
            t (int): timestamp
        """
        #initialise cost to 0 at curr t 
        self.costs.append(0)
        self.backorder_costs.append(0)
        self.ordering_costs.append(0)
        self.holding_costs.append(0)
        self.update_inventory(t)
        for demandable in self.upstream:
            demandable.update_all_inventory(t)

    def plot_cost(self):
        if not self.upstream:
            return
        df = pd.DataFrame(columns=["time", "cost", "type"])
        for i, val in enumerate(self.holding_costs):
            df.loc[len(df.index)] = [i, val, "holding cost"]
        for i, val in enumerate(self.backorder_costs):
            df.loc[len(df.index)] = [i, val, "backorder cost"]
        for i, val in enumerate(self.ordering_costs):
            df.loc[len(df.index)] = [i, val, "order cost"]
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.pointplot(data=df, x='time', y='cost', hue='type', ax=ax)
        # label points on the plot
        # for x, y in zip(df['time'], df['cost']):
        #     plt.text(x = x, y = y+10, s = "{:.0f}".format(y), color = "purple") 
        # # sns.relplot(kind='line', data=df, x='time', y='cost', hue='type')
        plt.show()
        for demandable in self.upstream:
            demandable.plot_cost()

    def plot_inv_level(self):
        if not self.upstream:
            return
        df = pd.DataFrame(columns=["time", "level", "item"])
        for i, dictionary in enumerate(self.inv_level_plot):
            for key, value in dictionary.items():
                df.loc[len(df.index)] = [i, value, key.get_name()[:9]]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax = sns.pointplot(data=df, x='time', y='level', hue='item', ax=ax)
        plt.setp(ax.collections, alpha=.3) #for the markers
        plt.setp(ax.lines, alpha=.3)       #for the lines
        plt.show()
        for demandable in self.upstream:
            demandable.plot_inv_level()


    def find_optimal_cost(self):
        curr_cost = 0
        if self.upstream:
            expected_holding_time = self.stochastic_lead_time.get_expected_value()
            curr_cost += expected_holding_time * self.holding_cost
            for item in self.inv_map:
                curr_cost += item.get_cost()
            for demandable in self.upstream:
                curr_cost += demandable.find_optimal_cost()
        return curr_cost

    def get_hc(self) -> int:
        """Returns holding cost for current demandable

        Returns:
            int: holding cost for current demandable
        """
        total = 0
        for item in self.inv_level:
            item_amt = self.inv_level[item]
            total += item_amt
        return total * self.holding_cost

    def get_total_cost(self) -> int: 
        """Returns total cost for all upstream demandable


        Returns:
            int: total cost incurred by all upstream demandable
        """
        total = self.total_costs
        for demandable in self.upstream:
            total += demandable.total_costs
        return total

    def get_curr_total_costs(self, t):
        total = self.get_curr_cost(t)
        for demandable in self.upstream:
            total += demandable.get_curr_cost(t)
        return total

    def get_curr_cost(self, t):
        """Retrieves total cost at specified time stamp for curr demandable

        Args:
            t (int): time stamp

        Returns:
            int: cost at specified time stamp
        """
        return self.costs[t]

    def update_all_cost(self, t):
        """Add hc into total cost

        Args:
            t (int): curr timestamp
        """
        self.holding_costs[t] += self.get_hc()
        self.total_costs += self.get_hc()
        self.backorder_costs[t] += self.backorder * self.backorder_cost
        self.total_costs += self.backorder * self.backorder_cost
        self.costs[t] = self.holding_costs[t] + self.backorder_costs[t] + self.ordering_costs[t]
        self.inv_level_plot.append(self.inv_level.copy())
        for demandable in self.upstream:
            demandable.update_all_cost(t)

    def print_upstream(self):
        name = [self.name]
        if self.upstream:
            for demandable in self.upstream:
                name += demandable.print_upstream()
        return name

    def print_upstream_cost(self, t):
        cost = [self.name, self.get_curr_cost(t)]
        if self.upstream:
            for demandable in self.upstream:
                cost += demandable.print_upstream_cost(t)
        return cost

    def print_inv_level(self):
        return "inv level: " + str(self.inv_level)

    def print_inv_pos(self):
        return "inv pos: " + str(self.inv_pos)

    def print_orders(self):
        s = ''.join([str(x) for x in self.arrivals])
        return "orders: " + s
    
    def print_total_cost(self):
        return "total cost: " + str(self.costs[max(0, len(self.costs) - 1)])

    def print_holding_cost(self):
        return "holding cost: " + str(self.holding_costs[max(0, len(self.holding_costs) - 1)])
    
    """ def print_ordering_cost(self):
        return "ordering cost: " + str(self.ordering_costs[max(0, len(self.ordering_costs) - 1)]) """
    
    def print_backorder_cost(self):
        return "backorder unit: " + str(self.backorder) + " backorder cost: " + str(self.backorder_costs[max(0, len(self.backorder_costs) - 1)])

    def print_inv_map(self):
        return "inv map: " + str(self.inv_map)
    
    def print_upstream_state(self):
        string = str(self)
        for demandable in self.upstream:
            string += "\n" + demandable.print_upstream_state()
        return string


    def __str__(self):
        return self.name + "\n" + self.print_inv_level() + "\n" + self.print_inv_pos() + "\n" + self.print_orders() + "\n" + self.print_total_cost() \
        + "\n" + self.print_holding_cost() + "\n" + self.print_backorder_cost() + "\n" + self.print_inv_map() 
    
    def __repr__(self):
        return "Demandable({})".format(self.name)

    