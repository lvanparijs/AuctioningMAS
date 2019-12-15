import numpy
import matplotlib.pyplot as plt

#DESCRIPTION
#This is a simulation of a modified second-price sealed-bid (Vickrey) auction

def vickrey_auction(k,n,r,s_max,e,lvl_commitment):
    # OUTPUTS
    market_price = 0  # Market price

    buyer_profit = numpy.zeros(n)  # Profit of each buyer
    buyer_paid = numpy.zeros(n)  # How much each buyer paid
    buyer_market_price = numpy.zeros(n)
    buy_from = -numpy.ones(n)  # Registers from which seller the buyer bought its item

    sellers_profit = numpy.zeros(k)  # Profits for each seller
    sellers_price = numpy.random.uniform(0, s_max, k)  # Random starting price for sellers
    sellers_id = numpy.arange(0, k)  # Ids for each seller, used to randomise the seller order each round

    bidding_factors = numpy.random.uniform(1.01, 2, n)  # Bidding , locked between 1 and 2
    bid_inc_factors = numpy.random.uniform(1.01, 2, n)  # Bid increase factors, randomly generated
    bid_dec_factors = numpy.random.uniform(0.01, 1, n)  # Bid decrease factors, randomly generated

    profits = numpy.zeros((n, r))  # Profits, for plotting purposes
    bid_fac = numpy.zeros((n, r))  # Bidding factors, also for plotting
    mps = numpy.zeros(r)

    if n > k: #more buyers than sellers necessary
        for rr in range(r): #Amount of rounds
            sellers_price = numpy.random.uniform(0, s_max/2, k) #Generate the starting prices of each seller randomly
            numpy.random.shuffle(sellers_id) #Ensures random order at every round
            out = [] #List of which buyers are not allowed to bid anymore, used for PURE auctioning
            reset(s_max,k,n) #Resets some variables between auctions

            print("===============")
            print("ROUND "+str(rr)) #Printing stuff
            print("===============")

            for kk in range(k): #Each seller has one auctions per round
                cur_seller_id = sellers_id[kk] #Gets the current seller ID, random due to shuffle function before each round
                start_price = sellers_price[cur_seller_id] #Get the starting price of the current seller
                bids = bidding_strategy(start_price, lvl_commitment, n, e, buy_from, buyer_market_price, buyer_paid, bidding_factors) #generate bids
                if len(out) > 0:
                    bids[out[:]] = -1 #If they have already won an auction, set bid to -1 so they dont participate
                    market_price = numpy.average(bids[bids >= 0])  # Only averages the actual bids, not the -inf
                else:
                    print(bids)
                    market_price = numpy.average(bids)

                market_price = numpy.average(bids[bids>=0]) #Only averages the actual bids, not the -inf

                print("AUCTION "+str(kk)+", organised by id: "+str(cur_seller_id))
                print("Starting price: "+str(start_price))
                print("Bids: "+str(bids))
                print("Market Price: "+str(market_price))
                print("===============")

                win_id, win_price = find_nearest_less_than(market_price,bids) #Get the highest bid lower than the market value, the buyers that bid higher than market price wont go through with it since it is irrational

                #Since the winner only needs to pay the second highest price the win price needs to be adapted
                _, win_price2 = find_nearest_less_than(win_price-0.00000001,bids) #Find second highest price

                if win_price2 == -1 or win_price2 >= win_price: #means there is only one valid bid below market value
                    win_price = (win_price+start_price)/2 #Average bid and starting price
                elif win_price2 <= win_price:
                    win_price = win_price2 #Set second highest price as winning price

                print("WINNING BID: (id)"+str(win_id)+", (bid)"+str(bids[win_id])+", (price)"+str(win_price))
                print("Bidding Factors: "+str(bidding_factors))

                profits[:,rr] = buyer_profit[:] #Adding profits after rr rounds to the table
                bid_fac[:,rr] = bidding_factors[:] #Adding bidding factors after rr rounds to the stable
                update_bidding_factors(bidding_factors, bids, market_price, out, win_id, n, bid_dec_factors, bid_inc_factors)

                if lvl_commitment: #Leveled Commitment, dont remove the winner
                    print("LEVELLED COMMITMENT AUCTIONING")
                    if buy_from[win_id] != -1: #If a buyer previously won an auction already
                        penalty_fee = 0
                        seller_id = -1
                        #Find lowest profit margin
                        old_profit = buyer_profit[win_id]
                        new_profit = market_price - win_price
                        if old_profit < new_profit:  # If new profit larger, decommit from old item
                            penalty_fee = e * buyer_paid[win_id]
                            seller_id = buy_from[win_id]
                            sellers_profit[int(seller_id)] -= (buyer_paid[win_id] - penalty_fee)  # Seller refunds buyer, but keeps fee
                            buyer_profit[win_id] -= penalty_fee
                            buyer_market_price[win_id] = market_price
                            buy_from[win_id] = cur_seller_id
                            buyer_paid[win_id] = win_price
                        else:
                            penalty_fee = e * win_price
                            seller_id = cur_seller_id
                            sellers_profit[int(seller_id)] -= (win_price - penalty_fee)  # Seller refunds buyer, but keeps fee
                            buyer_profit[win_id] -= penalty_fee
                        sellers_profit[int(seller_id)] += win_price
                        buyer_profit[win_id] += new_profit
                    else: #If this is the first auction this agent participates in
                        sellers_profit[cur_seller_id] += win_price
                        buyer_paid[win_id] = win_price
                        buyer_profit[win_id] += market_price - win_price
                        buyer_market_price[win_id] = market_price
                        buy_from[win_id] = cur_seller_id

                else: #Pure Auction, remove the winner from bidders
                    print("PURE AUCTIONING")
                    sellers_profit[cur_seller_id] += win_price
                    buyer_paid[win_id] = win_price
                    buyer_profit[win_id] += (market_price - win_price)
                    buyer_market_price[win_id] = market_price
                    buy_from[win_id] = cur_seller_id
                    out += [win_id]  # Add winner to out list
                mps[rr] = market_price

                print("SELLER PROFITS: " + str(sellers_profit))
                print("BUYER PROFITS: " + str(buyer_profit))
                print("BUYER BIDDING FACTORS"+str(bidding_factors))
                print("BUY FROM: " + str(buy_from))
    else:
        print("INSUFFICIENT BUYERS")
        print("The number of buyers(n) must always be greater than number of sellers(k)")
    plot_lines(profits,lvl_commitment,k,n,r,e)
    plot_market_price(mps,r)
    plot_lines(bid_fac,lvl_commitment,k,n,r,e)

def reset(s_max, k, n):
    sellers_price = numpy.random.uniform(0, s_max, k)  # Random starting price for sellers
    buyer_paid = numpy.zeros(n)  # How much each buyer paid
    buyer_market_price = numpy.zeros(n)
    #bidding_factors = numpy.random.uniform(1.01, 2, n)  # Bidding factors, UNCOMMENT IF YOU WANT THE BIDDING FACTOR THE BE RESET AFTER EACH ROUND
    #bid_inc_factors = numpy.random.uniform(1.01, 2, n)  # Bid increase factors, randomly generated
    #bid_dec_factors = numpy.random.uniform(0.01, 1, n)  # Bid decrease factors, randomly generated

def bidding_strategy(start_price, lvl_commitment, n, e, buy_from, buyer_market_price, buyer_paid, bidding_factors):
    if lvl_commitment:
        v = numpy.zeros(n)
        for i in range(n):
            if buy_from[i] != -1:  # If this buyer has bought something before
                v[i] = ((buyer_market_price[i] - buyer_paid[i]) + e * buyer_paid[i])
        res = bidding_factors * start_price - v
        res[res < 0] = 0
        return res
    else:
        return bidding_factors*start_price

def update_bidding_factors(bidding_factors, bids, market_price, out, winner_id, n, bid_dec_factors, bid_inc_factors): #Updates the bidding factors
    ids = numpy.arange(0, n)
    if len(out) > 0: #if there have not been any bids yet, this is not neccesary
        numpy.delete(ids,out)
    for id in ids:#update bidding factors
        if id == winner_id or bids[id] >= market_price: #If this bidder won the last auction or if it bid too much
            bidding_factors[id] = bidding_factors[id]*bid_dec_factors[id]
        else:
            bidding_factors[id] = bidding_factors[id]*bid_inc_factors[id]
    bidding_factors[bidding_factors<1] = 1
    bidding_factors[bidding_factors>2] = 2

def find_nearest_less_than(search_val, input_data):
    diff = input_data - search_val
    diff[diff>=0] = -1
    idx = diff.argmax()
    return idx, input_data[idx]

def plot_market_price(mp, r):
    plt.figure()
    plt.title("Market Price over "+str(r)+" rounds.")
    rounds = numpy.linspace(0, r, r)
    plt.plot(rounds, mp[:])
    plt.show()

def plot_lines(factors, lvl_commitment, k, n, r, e):
    plt.figure()
    if lvl_commitment:
        plt.title("Profits by "+str(n)+" agents w/"+str(k)+" sellers over "+str(r)+" rounds. pen = "+str(e)+" (LVLD COM)")
    else:
        plt.title("Profits by " + str(n) + " agents w/" + str(k) + " sellers over " + str(r) + " rounds. pen = "+str(e)+" (PURE AUC)")
    rounds = numpy.linspace(0, r, r)
    for i in range(factors.shape[0]):
        plt.plot(rounds,factors[i,:])
    plt.show()


print("k = num of Sellers, n = num of Buyers, r = num of Rounds, smax = Max starting price, e = penalty factor, lvl = True/False use leveled commitment")
print("Input the variables one by one, fill in variable and press ENTER")
print("int, int, int, float, float, bool // Example input: 5 [ENTER] 4 [ENTER] 100 [ENTER] 1. [ENTER] 0.4 [ENTER] True [ENTER] ")
print("Please enter input in order: k, n, r, smax, e, lvl \n")
vars = []
for i in range(0,6):
    var = input()
    if i < 3:
        vars += [int(var)]
    elif i < 5:
        vars += [float(var)]
    else:
        vars += [bool(var)]
#.split(' ')
print(vars)

vickrey_auction(vars[0],vars[1],vars[2],vars[3],vars[4],vars[5])