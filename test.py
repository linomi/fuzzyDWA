import numpy as np 
from matplotlib import pyplot as plt
from numpy import sin,cos,pi,sqrt,arctan2,sign,abs
from time import sleep
from matplotlib.pyplot import axis
from matplotlib import pyplot as plt
import numpy as np 
import time 
import math 
import bezier
from pyswarm import pso as ps
from fuzzylogic.classes import Domain,Set,Rule
from fuzzylogic.functions import triangular,trapezoid,S,R
from fuzzylogic.classes import rule_from_table


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def angle_range_corrector(angle):
    if angle > np.pi:
        while angle > np.pi:
            angle -= 2 * np.pi
    elif angle < -np.pi:
        while angle < -np.pi:
            angle += 2 * np.pi

    return angle

def endpoint_on_circul(w,v,x_0,y_0,phi_0,steps):
    steps = steps
    X = []
    Y = []
    x = x_0
    y = y_0
    phi = phi_0
    if w!=0:
        for i in range(steps):
            x = x + (v/w)*(sin(phi+w*1/steps) - sin(phi))
            y = y - (v/w)*(cos(phi +w*1/steps) - cos(phi))
            phi = phi + w*0.01
            X.append(x)
            Y.append(y)
    else:
         for i in range(steps):  
            x =x + v*cos(phi)
            y =y + v*sin(phi)
            X.append(x)
            Y.append(y)
            phi = phi
    phi = angle_range_corrector(phi)
    return X,Y,x,y,phi
    
def dist(w,v,x_temp,y_temp,phi_temp,obstacles_temp):
    pre_step = 20
    min_dist= 1e9
    X,Y,x,y,theta= endpoint_on_circul(w,v,x_temp,y_temp,phi_temp,pre_step)
    for obstacle in obstacles_temp:
        for _ in range(0,pre_step):
            d = sqrt((obstacle[0]-X[_])**2 + (obstacle[1]-Y[_])**2)
            if d < min_dist:
                min_dist = d
    return X,Y,min_dist,x,y,theta   
def admissible_v(vmin,vmax,wmin,wmax,x_now,y_now,phi_now,obstacles_now):
    res = 30
    V = np.linspace(vmin,vmax,res)
    W = np.linspace(wmin,wmax,res)
    admiss = []
    #plt.box([vmin,vmax],[wmin,wmax])
    for i in range(0,res):
        for j in range(0,res):
            X,Y,d,x,y,phi= dist(W[j],V[i],x_now,y_now,phi_now,obstacles_now)
            if V[i]<= 3*sqrt(2*d*0.2) and W[i]<= 3*sqrt(2*d*0.3):
                admiss.append([V[i],W[j],X,Y,d,x,y,phi])
            else:
                #plt.plot(X,Y,'r')
                pass
    return admiss

#class for calculating global path
class PSO_bezier():
    def __init__(self,start,end,number_of_nodes,constraints) -> None:
        self.start = start 
        self.end = end 
        self.N = number_of_nodes
        self.constraints = constraints
        self.solution =  np.array([np.random.uniform(self.start[0],self.end[0],self.N) ,np.random.uniform(self.start[1],self.end[1],self.N)])
        pass
    def bezier_curve_cost(self,p):
      #print(p.shape)
      p = np.array([np.split(p,2)[0],np.split(p,2)[1]])
      p = np.insert(p,0,self.start,axis = 1)
      p = np.insert(p,len(p.transpose()),self.end,axis = 1)
      self.curve = bezier.Curve(p,degree= self.N)
      cost = self.curve.length 
      return cost
    def con(self,x):
      c = self.curve.evaluate_multi(np.linspace(0,1,100)).transpose()
      cost = 0
      for k in c: 
        for constrain in self.constraints:
          if (k[0]-constrain[0])**2 +(k[1]-constrain[1])**2<constrain[2]**2:
            cost = 100000
    
      return -cost
    def PSO(self):
        lb = [-100 for i in range(0,2*self.N-2)]
        ub = [100 for i in range(0,2*self.N-2)]
        s =ps(self.bezier_curve_cost,lb,ub,f_ieqcons=self.con) 
        return s

x_start = 0 
y_start = 0
theta_start = 0
target_x = 15
target_y = 15


# calculating global path
start = np.array([x_start,y_start])
stop = np.array([target_x,target_y])
number_of_nodes = 4
walking_steps = 100
constrain = np.array([[5,5,1],[5,4,1],[12,10,1.2],[8,10,1],[8,12,1.2],[2,1,1]])
pso = PSO_bezier(start,stop,number_of_nodes,constrain)
s,l = pso.PSO()
s = np.array([np.split(s,2)[0],np.split(s,2)[1]])
s = np.insert(s,0,start,axis = 1)
s = np.insert(s,len(s.transpose()),stop,axis = 1)
curve = bezier.Curve(s,degree=number_of_nodes)
path = curve.evaluate_multi(np.linspace(0,1,walking_steps))
#plt.plot(path[1],path[0])
fig,ax = plt.subplots()
for c in constrain:
  circle = plt.Circle((c[0],c[1]),radius=c[2],color = 'green')
  ax.add_patch(circle)
#plt.plot(s[0],s[1],color ='green')
plt.plot(path[0],path[1],color='r')
plt.scatter([x_start,target_x],[y_start,target_y],marker='x',c='blue')
plt.xlim(-1,17)
plt.ylim(-1,17)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#fuzzy inference system 
kapa = Domain('Collision Risk',-1,1,res= 0.01)
mu = Domain('dist to dynamic obstacle',0,5,res=0.01)
beta = Domain('coef of heading',0,3,res=0.01)
gamma = Domain('coef of loyalty',0,3,res= 0.01)

fig = plt.figure(figsize=(20,15))
kapa.S = S(0,0.1)
kapa.RS =  triangular(0,.3)
kapa.N = triangular(0.1,0.4)
kapa.RB = triangular(0.3,0.5)
kapa.B = R(0.4,0.5)
plt.subplot(2,2,1)
kapa.S.plot()
kapa.RS.plot()
kapa.N.plot()
kapa.RB.plot()
kapa.B.plot()
plt.legend(['S','RS','N','RB','B'])
mu.DS = S(0.5,1.5)
mu.DN = triangular(0.5,2)
mu.DF = R(1.5,2)
plt.subplot(2,2,2)
mu.DS.plot()
mu.DN.plot()
mu.DF.plot()
plt.legend(['DS','DN','DF'])

beta.L = S(0,1.5)
beta.M = triangular(1,2)
beta.H = R(1.5,3)
plt.subplot(2,2,3)
beta.L.plot()
beta.M.plot()
beta.H.plot()
plt.legend(['L','M','H'])
gamma.L = trapezoid(0,0.5,0.5,1.5)
gamma.M = triangular(1,2)
gamma.H = trapezoid(1.5,2.5,2.5,3)
plt.subplot(2,2,4)
gamma.L.plot()
gamma.M.plot()
gamma.H.plot()
plt.legend(['L','M','H'])
plt.show()
plt.clf()
plt.cla()
#rule for beta 
rule_beta = '''
        kapa.S      kapa.RS     kapa.N      kapa.RB     kapa.B
mu.DS   beta.H      beta.H      beta.H      beta.H      beta.H    
mu.DN   beta.H      beta.H      beta.M      beta.M      beta.L
mu.DF   beta.H      beta.H      beta.M      beta.L      beta.L
'''
beta_rule = rule_from_table(rule_beta,globals())

#rule for gamma
rule_gamma = '''
        kapa.S      kapa.RS     kapa.N      kapa.RB     kapa.B
mu.DS   gamma.L     gamma.L     gamma.L     gamma.L     gamma.L    
mu.DN   gamma.L     gamma.L     gamma.M     gamma.M     gamma.H
mu.DF   gamma.L     gamma.L     gamma.M     gamma.H     gamma.H
'''
gamma_rule = rule_from_table(rule_gamma,globals())



#dwa and local path planing
theta_start = pi/4
x = x_start
y = y_start
theta = theta_start
max_acc = 1
max_angacc = 0.3
w_chosen = 0
v_chosen =0 
XX = []
YY = []
time_step = 0.1
#defining dynamic obstacle
dynamic_obstacle = [[4,10]]
#for k in range(0,500):
k = 0 
fig,ax = plt.subplots()
while sqrt((x-target_x)**2 + (y-target_y)**2)>0.2:
    if w_chosen !=0:
        x = x + (v_chosen/w_chosen)*(sin(theta +w_chosen*time_step)- sin(theta))
        y = y - (v_chosen/w_chosen)*(cos(theta +w_chosen*time_step)- cos(theta))
        theta = theta + w_chosen*time_step
    else:
        x =x + v_chosen*cos(theta)*time_step
        y =y + v_chosen*sin(theta)*time_step
    theta = angle_range_corrector(theta)
    XX.append(x)
    YY.append(y)
    #plt.plot(XX,YY,'r')
    #plt.pause(0.8)
    #plt.cla()
    vmax = v_chosen + 2*time_step
    #if vmax>1:
    vmax = 1
    vmin = v_chosen - 2*time_step
    vmin = vmin/2*(sign(vmin)+1) +1
    wmax = w_chosen + 3*time_step
    #if wmax>0.3:
    #    wmax = 0.3
    wmin = w_chosen - 3*time_step 
    #if wmin<-0.3:
    #    wmin = -0.3 
    dynamic_obstacle[0][0] = 10 -2*sin(k/10)
    dynamic_obstacle_velocity_vector = np.array([-0.2*cos(k/10),0])
    robot_velocity_vector = np.array([v_chosen*cos(theta),v_chosen*sin(theta)])
    
    v_RO = robot_velocity_vector - dynamic_obstacle_velocity_vector
    RO = np.array([-x+dynamic_obstacle[0][0],-y+dynamic_obstacle[0][1]])
    theta_c = angle_between(RO,v_RO)
    muu = np.linalg.norm(RO)
    kapaa = np.linalg.norm(v_RO)*cos(theta_c)
    fis_input = {mu:muu,kapa:kapaa}
    betaa = beta_rule(fis_input)
    gammaa = gamma_rule(fis_input)   
    k += 1
    search_space = admissible_v(vmin,vmax,wmin,wmax,x,y,theta,dynamic_obstacle)
    G = []

    for point in search_space:
        vel = point[0]
        dis = point[4]
        heading =2*pi -(abs(point[7] -(arctan2((target_y - point[6]),(target_x - point[5])))))
        loyalty = 1e5
        for p in range(0,100):
            dx = path[0,p] - point[5]
            dy = path[1,p] - point[6]
            loyalty_p = sqrt(dx**2 +dy**2)
            if loyalty_p <loyalty:
                loyalty = loyalty_p
        loyalty = - loyalty 
        cost = 0.05*heading + 0.2*vel + betaa*dis  + 2*gammaa*loyalty
        G.append(cost)
    g = np.array(G)
    arguman = np.argmax(g)
    best_selction = search_space[arguman]
    #print(best_selction)
    plt.plot(target_x, target_y,'bo')
    for ob in dynamic_obstacle:
        plt.plot(ob[0],ob[1],'yx')
    plt.plot(XX,YY,'r')
    plt.xlim((0,17))
    plt.ylim((0,17))
    plt.plot(best_selction[2],best_selction[3],'--g')
    plt.text(x,y,str(k))
    #plt.plot(s[0],s[1],color ='green')
    plt.plot(path[0],path[1],color='black') 
    for c in constrain:
        circle = plt.Circle((c[0],c[1]),radius=c[2],color = 'green')
        ax.add_patch(circle)
    file_name = f'gif/{k}.png'
    plt.savefig(file_name)
    plt.cla()
    v_chosen = best_selction[0]
    w_chosen = best_selction[1]
#plt.plot(XX,YY)
plt.show()