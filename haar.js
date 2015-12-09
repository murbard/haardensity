// inefficient boxmuller 
function randn() {
    u1 = Math.random(); u2 = Math.random();
    return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI * u2);
}
// cdf of normal distribution
function cdf(x, mean, variance) {
  return 0.5 * (1 + erf((x - mean) / (Math.sqrt(2 * variance))));
}

// uses this erf implementation
function erf(x) {
  // save the sign of x
  var sign = (x >= 0) ? 1 : -1;
  x = Math.abs(x);

  // constants
  var a1 =  0.254829592;
  var a2 = -0.284496736;
  var a3 =  1.421413741;
  var a4 = -1.453152027;
  var a5 =  1.061405429;
  var p  =  0.3275911;

  // A&S formula 7.1.26
  var t = 1.0/(1.0 + p*x);
  var y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y; // erf(-x) = -erf(x);
}


function SimpleModel(points, sparsity) {    
    this.sparsity = sparsity;
    this.n = points.length;
    this.Q = new Quad();    
    for(var i = 0; i < this.n; ++i)
      this.Q.insert(points[i], 1);                 
}    


SimpleModel.prototype.pixels = function(size) {
    var pixels = [];
        
    var max = -Infinity;
    var min =  Infinity;
    for(var i = 0; i < size; ++i) {
        pixels[i] = [];
        for(var j = 0; j < size; ++j)  {
            pixels[i][j] = 0;
            for(var di = 0; di < 1; ++di) {
                for(var dj = 0; dj < 1; ++dj) {
                    var x = (4*i+2*di+1)/(4*size);
                    var y = (4*j+2*dj+1)/(4*size);                    
                    for(var y = (j+0.25)/size; y<(j+1)/size; y += 0.5/size)                    
                            pixels[i][j] += this.Q.pdf(this.sparsity, {x:x, y:y});                     
                }
            }
            max = Math.max(max, pixels[i][j]);
            min = Math.min(min, pixels[i][j]);
        }
    }
    return {pixels:pixels, min:min, max:max};
}

    
    

function Model(points, sparsity) {

    this.sparsity = sparsity;
    // first do a PCA of the point set
    this.n = points.length;

    this.vx = 0; this.vy = 0; this.cv = 0;
    this.mx = 0; this.my = 0;

    for(var i = 0; i < this.n; ++i) {
       this.vx += points[i].x * points[i].x;
       this.vy += points[i].y * points[i].y;
       this.cv += points[i].x * points[i].y;
       this.mx += points[i].x;
       this.my += points[i].y;
    }
    this.mx /= this.n; this.my /= this.n;
    this.vx = this.vx / this.n - this.mx * this.mx;
    this.vy = this.vy / this.n - this.my * this.my;
    this.cv = this.cv / this.n - this.mx * this.my;

   // cholesky decomposition
   chol = [[1/Math.sqrt(this.vx), 0],[this.cv/Math.sqrt(this.vx*(this.vx*this.vy-this.cv*this.cv)), -this.vx/Math.sqrt(this.vx*(this.vx*this.vy-this.cv*this.cv))]];
   this.chol = chol;

   this.pca_points = [];
   for(var i = 0; i < this.n; ++i)
      this.pca_points.push({x: chol[0][0] * (points[i].x - this.mx), y: chol[1][0] * (points[i].x - this.mx) + chol[1][1] * (points[i].y - this.my)});
   
   this.transforms = [];
}


// draw a random transformation and compute the evidence
Model.prototype.draw = function() {

   
      var theta = Math.random() * 2*Math.PI;
      var  phi = Math.random() * Math.PI / 2; // just a reminder that there's a quarter turn symmetry anyway
      var sx    = randn() * Math.sqrt(1/(2*this.n)) + Math.sqrt(1 + 1/(4*this.n)); // no need to pick a sign, Haar is symmetric
      var sy    = randn() * Math.sqrt(1/(2*this.n)) + Math.sqrt(1 + 1/(4*this.n));
   
 
   var M = [[sx*Math.cos(phi)*Math.cos(theta)-sy*Math.sin(phi)*Math.sin(theta), -sx*Math.cos(theta)*Math.sin(phi)-sy*Math.cos(phi)*Math.sin(theta)],
            [sy*Math.cos(theta)*Math.sin(phi)+sx*Math.cos(phi)*Math.sin(theta), sy*Math.cos(phi)*Math.cos(theta) - sx*Math.sin(phi)*Math.sin(theta)]];

   var tx = randn() * Math.sqrt(1/this.n), ty = randn() * Math.sqrt(1/this.n);
   Q = new Quad();
   for(var i = 0; i < this.n; ++i)
      Q.insert({ x: cdf(M[0][0] * this.pca_points[i].x + M[0][1] * this.pca_points[i].y + tx, 0, 1),
                 y: cdf(M[1][0] * this.pca_points[i].x + M[1][1] * this.pca_points[i].y + ty, 0, 1) }, 1)

   var A = [[M[0][0] * this.chol[0][0] + M[0][1] * this.chol[1][0], M[0][0] * this.chol[0][1] + M[0][1] * this.chol[1][1]],
             [M[1][0] * this.chol[0][0] + M[1][1] * this.chol[1][0], M[1][0] * this.chol[0][1] + M[1][1] * this.chol[1][1]]];
   var b = [tx - A[0][0] * this.mx  - A[0][1] * this.my, ty - A[1][0] * this.mx - A[1][1] * this.my];
   var detA = Math.abs(A[0][0]*A[1][1]-A[0][1]*A[1][0]);
         
   this.transforms.push( {theta:theta, phi:phi, sx:sx, sy:sy, tx:tx, ty:ty, A:A, b:b, evidence:Q.marginalize(this.sparsity), Q:Q, detA:detA} )
}


Model.prototype.pixels = function(size) {
    var pixels = [];
    
    var maxEvidence = -Infinity;
    for(var t = 0; t < this.transforms.length; ++t) 
        maxEvidence = Math.max(this.transforms[t].evidence, maxEvidence);    
    
    var max = -Infinity;
    var min =  Infinity;
    for(var i = 0; i < size; ++i) {
        pixels[i] = [];
        for(var j = 0; j < size; ++j)  {
            pixels[i][j] = 0;
            for(var di = 0; di < 1; ++di) {
                for(var dj = 0; dj < 1; ++dj) {
                    var x = (4*i+2*di+1)/(4*size);
                    var y = (4*j+2*dj+1)/(4*size);                    
                    for(var y = (j+0.25)/size; y<(j+1)/size; y += 0.5/size)        
                        for(var t = 0; t < this.transforms.length; ++t) {
                            var trans = this.transforms[t];
                            var xx = trans.A[0][0] * x + trans.A[0][1] * y + trans.b[0];
                            var yy = trans.A[1][0] * x + trans.A[1][1] * y + trans.b[1];                        
                            pixels[i][j] += trans.detA * Math.exp(
                                trans.evidence - maxEvidence 
                                - 0.5 * (xx*xx+yy*yy))* 
                                    trans.Q.pdf(this.sparsity, {x:cdf(xx,0,1), y:cdf(yy,0,1)}); 
                        }
                }
            }
            max = Math.max(max, pixels[i][j]);
            min = Math.min(min, pixels[i][j]);
        }
    }
    return {pixels:pixels, min:min, max:max};
}






/*
NN.prototype.sigma = function(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

NN.prototype.logdsigma = function(x) {
    return (x>0)?(-x-2*Math.log(1+Math.exp(-x))):(x-2*Math.log(1+Math.exp(x)));
}


NN.prototype.jacobian = function(point) {
    var x = this.M[0] * (point.x + this.b[0]) + this.M[1] * (point.y + this.b[1]);
    var y = this.M[2] * (point.x + this.b[0]) + this.M[3] * (point.y + this.b[1]);
    return {
       p: { x:this.sigma(x), y:this.sigma(y)},
       j: this.jacT + this.logdsigma(x) + this.logdsigma(y) };
}*/

// a quad tree
function Quad() {
    this.n = 0; 
    this.p = null;
    this.children = null;
}

Quad.prototype.insert_deep = function(point, n) {
    if(point.y < 0.5) {
        if(point.x < 0.5)
            this.children[0].insert({x:2*point.x, y:2*point.y}, n);
        else
            this.children[1].insert({x:2*point.x-1, y:2*point.y}, n);
    } else {
        if(point.x < 0.5)
            this.children[2].insert({x:2*point.x, y:2*point.y-1}, n);
        else
            this.children[3].insert({x:2*point.x-1, y:2*point.y-1}, n);
    }
}

Quad.prototype.depth = function()
{
    if (this.children)
        return 1+Math.max(this.children[0].depth(),this.children[1].depth(),this.children[2].depth(),this.children[3].depth());
    return 1;
}

Quad.prototype.insert = function(point, m) {
    this.n += m;
    if (this.children) {
       this.insert_deep(point,m);
       return;
    }
    if (!this.p) {
       this.p = {x:point.x, y:point.y};       
       return;
    }
    if (this.p.x==point.x && this.p.y==point.y)
       return;
    
    this.children = [new Quad(), new Quad(), new Quad(), new Quad()];
    this.insert_deep(this.p, this.n-m);
    this.insert_deep(point, m);
    this.p = null;
}


function B(L) {
    var f = 1.0;
    var c = 1.0;
    for (var k=0;k< L.length;++k) {
        f /= (L[k]+1);
        for (var i = 1; i <= L[k]+1; ++i, ++c)
            f *= i / c;
    }    
    return f*(c-1);
}

function Blah(P,Q,g) {
   var sum = 0.0;
   var i = 0;
   var j = 0;
   var m = 1;
   var n = 1;
   var c = 0.0;
   
   while(i < P.length || j < Q.length) {
      var y = Math.log(m/n*g) - c;
      var t = sum + y;
      c = (t-sum)-y;
      sum = t;      
      if(i<P.length) {m++; if (m>P[i]) { i++; m=1}}
      if(j<Q.length) {n++; if (n>Q[j]) { j++; n=1}}      
   }
   return Math.exp(sum);
}

var Facto = []
Facto[0]=0;
for(var i =1; i < 1000000;++i)
   Facto[i] = Facto[i-1] + Math.log(i); 

function drawGamma(n) {
    var r = 1;
    for (var i = 0; i < n; ++i)
        r *= (1-Math.random());
    return -Math.log(r);
}

function drawDirichlet(X,n) {
    var g = [];
    var s = 0;
    for (var i=0;i<n;++i)
        s += g[i] = drawGamma(X[i] + 1);
    for (var i=0;i<n;++i)
        g[i] /= s;
    return g;
}


function OneBreak(w) {
  var y = Math.exp(-w);
  return (5+y*(20+y*(20+8*y)))/(20+y*(60+y*(60+20*y)));
}

function OnePdf(w, pa, pb) {
   var ia = 2*Math.round(pa.x)+Math.round(pa.y);
   var ib = 2*Math.round(pb.x)+Math.round(pb.y);
   if(ia==ib)  // same, need to divide
       return 4 * OneBreak(w) * OnePdf(w, {x:2*pa.x%1,y:2*pa.y%1},{x:2*pb.x%1,y:2*pb.y%1});   
   return 4*(1.0-OneBreak(w))/3.0;
}

Quad.prototype.clear = function()
{
  this.pp = null;
  if(this.children) 
     for(var k = 0; k < 4; ++k)  
        this.children[k].clear();
  
  
}

Quad.prototype.pdf = function(w, point) {
   if (!this.children) {
      if(this.n==0)
         return 1.0;
     return OnePdf(w,point, this.p);
   }   
   // there are children, so compute [p,q,r,s]
   // and recursively call pdf in the right child

   
   if (!this.pp) {
 
      var zz = this.getZ(w);
      var Z = zz[0];
      var meanp = this.getMeanP(w);
      var p = [0, 0, 0, 0];
    
     for(var i = 0; i < 8; ++i ) {     
        for(var k = 0; k < 4; ++ k)
           p[k] += Z[i] * meanp[i][k];
     }
     this.pp = p; 
   }
         
   if(point.y<0.5) {
     if (point.x<0.5)
        return  this.pp[0] * 4*this.children[0].pdf(w,{x:2*point.x, y:2*point.y});
     else 
       return  this.pp[1] * 4*this.children[1].pdf(w,{x:2*point.x-1, y:2*point.y});
   } else {
	if (point.x<0.5)
		return  this.pp[2] * 4*this.children[2].pdf(w,{x:2*point.x, y:2*point.y-1});
	else 
		return  this.pp[3] * 4* this.children[3].pdf(w,{x:2*point.x-1, y:2*point.y-1});
   }
}

Quad.prototype.getZ = function(w) 
{
    var sumZ = 0.0;
    var Z = [];
    var a = this.children[0].n;
    var b = this.children[1].n;
    var c = this.children[2].n;
    var d = this.children[3].n;
    var n = this.n;
    Z[0] = 0.0;
    Z[1] = Facto[a+b] + Facto[c+d] - Facto[n+1] + Math.log(2) * n - w ;
    Z[2] = Facto[a+c] + Facto[b+d] - Facto[n+1] + Math.log(2) * n - w;
    Z[3] = Facto[a+d] + Facto[b+c] - Facto[n+1] + Math.log(2) * n - w;
    Z[4] = Facto[a]+Facto[b]+Facto[c]+Facto[d] - Facto[a+b+1]-Facto[c+d+1] + Math.log(2)*n - 2*w; 
    Z[5] = Facto[a]+Facto[b]+Facto[c]+Facto[d] - Facto[a+c+1]-Facto[b+d+1] + Math.log(2)*n - 2*w;
    Z[6] = Facto[a]+Facto[b]+Facto[c]+Facto[d] - Facto[a+d+1]-Facto[b+c+1] + Math.log(2)*n - 2*w;
    Z[7] = Math.log(6)+Facto[a]+Facto[b]+Facto[c]+Facto[d]-Facto[n+3]-3*w + Math.log(4)*n;
 
    var largest = -Infinity;
    for(var i = 0; i < 8; ++i ) {
        if (Z[i] > largest)
		largest = Z[i];
}
    for(var i = 0; i < 8; ++i )
	sumZ += Z[i] = Math.exp(Z[i]-largest);
    for(var i = 0; i < 8; ++i )
	Z[i] /= sumZ;
    return [Z, sumZ, largest]
}

Quad.prototype.getMeanP = function(w)
{
    var a = this.children[0].n;
    var b = this.children[1].n;
    var c = this.children[2].n;
    var d = this.children[3].n;
    var n = this.n;
    var meanp = [];
     meanp[0] = [0.25,0.25,0.25,0.25];
     ab_vs_cd = (a + b + 1)/(n+2) ;
     meanp[1] = [ab_vs_cd/2, ab_vs_cd/2, (1-ab_vs_cd)/2, (1-ab_vs_cd)/2];
     ac_vs_bd = (a + c + 1)/(n+2);
     meanp[2] = [ac_vs_bd/2, (1-ac_vs_bd)/2, ac_vs_bd/2, (1-ac_vs_bd)/2];
     ad_vs_bc = (a + d+1)/(n+2);
     meanp[3] = [ad_vs_bc/2, (1-ad_vs_bc)/2, (1-ad_vs_bc)/2, ad_vs_bc/2];
     a_vs_b = (a +1) / (a + b + 2);
     c_vs_d = (c + 1) / (c + d + 2);
     meanp[4] = [a_vs_b/2, (1-a_vs_b)/2,c_vs_d/2, (1-c_vs_d)/2];
     a_vs_c = (a+1) / (a + c+2);
     b_vs_d = (b+1) / (b + d+2);
     meanp[5] = [a_vs_c/2, b_vs_d/2, (1-a_vs_c)/2, (1-b_vs_d)/2];
     a_vs_d = (a+1) / (a + d+2);
     b_vs_c = (b+1) / (b + c+2);
     meanp[6] = [a_vs_d/2,b_vs_c/2,(1-b_vs_c)/2,(1-a_vs_d)/2];
     meanp[7] = [(a+1)/(n+4), (b+1) /(n+4), (c+1) / (n+4), (d+1)/(n+4)];
     return meanp;

 
}


Quad.prototype.marginalize = function(w) {
    if (!this.children)
    {
       if(this.n>1)
          console.log("problem", this.n);
       return 0.0;
    }

    var zz = this.getZ(w);
    var Z = zz[0], sumZ = zz[1], largest = zz[2]; 

    var sumY = 1 +  3*Math.exp(-w) + 3*Math.exp(-2*w) + Math.exp(-3*w);    
    var Y = [1.0/sumY, Math.exp(-w) / sumY, Math.exp(-w)/sumY,Math.exp(-w)/sumY, Math.exp(-2*w)/sumY, Math.exp(-2*w)/sumY, Math.exp(-2*w)/sumY, Math.exp(-3*w)/sumY];
    
     
    var cub = Math.log(sumZ/sumY)+largest;

    
    for(var k = 0; k < 4; ++k)
        cub += this.children[k].marginalize(w);
    return cub;
}

PIXELS = [];
for(var x = 0; x < 512; ++x) {
   PIXELS[x] = [];
   for(var y = 0; y < 512; ++y) {
          PIXELS[x][y] = 0.0;
}
}

Quad.prototype.pixels = function(size,  w)
{
   //nn = new NN({tX:-0.523674632,tY:-0.4503297986451,phi:0,theta:0,sX:2.1218760347,sY:2.16556414});
   
   
   for(var x = 0; x < size; ++x) {           
      for(var y = 0; y < size; ++y) {
          PIXELS[x][y] = 0.0;
                jp = nn.jacobian({x:(x+0.25)/size,y:(y+0.25)/size});
          PIXELS[x][y] += 0.25*      Math.exp(jp.j)*this.pdf(w,jp.p);
                jp = nn.jacobian({x:(x+0.75)/size,y:(y+0.25)/size});
                PIXELS[x][y] += 0.25* Math.exp(jp.j)*this.pdf(w,jp.p);
                jp = nn.jacobian({x:(x+0.25)/size,y:(y+0.75)/size});
                PIXELS[x][y] += 0.25* Math.exp(jp.j)*this.pdf(w,jp.p);
                jp = nn.jacobian({x:(x+0.75)/size,y:(y+0.75)/size});
                PIXELS[x][y] += 0.25* Math.exp(jp.j)*this.pdf(w,jp.p);
      }
   }
   return PIXELS;
}




MCMC = function(trans, step) {
    this.trans = trans;
    this.chain = [];
    this.step = step;
    this.ll = -1e99;
    this.unique = 0;
}


MCMC.prototype.draw = function() {

   var trans = { 
      tX:    this.trans.tX + this.step * boxmuller(),
      tY:    this.trans.tY + this.step * boxmuller(),
      phi:   (this.trans.phi + this.step * boxmuller())%2*Math.PI,
      theta: (this.trans.phi + this.step * boxmuller())%2*Math.PI,
      sX:    this.trans.sX + this.step * boxmuller(),
      sY:    this.trans.sY + this.step * boxmuller() };    
    var n2 = new NN(trans);

    var total_j = 0.0;
    Q = new Quad();
    for(var i = 0; i < 1000; ++i) {
        var j = n2.jacobian(Points[i]);        
        Q.insert(j.p, 1);
        total_j += j.j;
    }
    total_j += Q.marginalize(2.3);
    if (Math.random() < Math.exp(total_j - this.ll)) {
        this.trans = trans;
        this.ll = total_j;
        this.unique++;
    }
    this.chain.push(this.trans);
}

Points = [];
for(var k = 0; k < 10000; ++k)
{
    var theta = Math.random() * 2 * Math.PI;
    Points.push({x:(1.5+Math.cos(theta))/3 + 0.05*randn(), y:(1.5+Math.sin(theta))/3+ 0.05*randn()});
}



//var mcmc = new MCMC(new NN([1.0,0,0,1.0],[0.0,0.0]), 0.1);
//for(var n = 0; n < 1; ++n)
//    mcmc.draw();

/*

//      var EPSILON = 1/10000000000; //CONST

        //checks if it is symmetric
        function isSymmetric(A){
                var N = A.length;
                for(var i=0;i<N;i++)
                {
                        for(var j =0; j<i;j++)
                        {
                                if (Math.abs(A[i][j] - A[j][i])/(Math.abs(A[i][j])+Math.abs(A[j][i]))>1e-10) {
                                        return false;
                                }
                        }
                }
                return true;
        }

        function isSquare(A){
        var N = A.length;
        for (var i = 0; i < N; i++) {
            if (A[i].length != N) {return false;}
        }
        return true;
    }
/////////////////////////////////////converted til here////////////////////

    // return Cholesky factor L of psd matrix A = L L^T
    function choleskyFactor(A){
        if (!isSquare(A)) {
            throw "RuntimeException: Matrix is not square";
        }
        if (!isSymmetric(A)) {
            throw "RuntimeException: Matrix is not symmetric";
        }

        var N  = A.length;
        var L = new Array(); //set every element to an array later to make 2D

        for (var i = 0; i < N; i++)  {
                        L[i] = new Array();
            for (var j = 0; j <= i; j++) {
                var sum = 0.0;
                for (var k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j){
                                        L[i][i] = Math.sqrt(A[i][i] - sum);
                                }
                else{
                                L[i][j] = 1.0 / L[j][j] * (A[i][j] - sum);
                                }
            }
            if (L[i][i] <= 0) {
                throw "RuntimeException: Matrix not positive definite";
            }
        }
        return L;
    }

COV = [[10,0,0,0,0,0],[0,10,0,0,0,0],[0,0,10,0,0,0],[0,0,0,10,0,0],[0,0,0,0,10,0],[0,0,0,0,0,10]];
MEAN = [0,0,0,0,0,0];



*/


/*
function drawRandomNormal() {

    c = choleskyFactor(COV);
    var norm = [];
    for(var k = 0; k<6;++k) {
        norm[k] = boxmuller();
    }
    var v  = [];
    for(var j = 0; j<6;++j) {
        v[j] = MEAN[j];
        for(var k = 0; k<=j;++k) {
            v[j] += c[j][k] * norm[k];
        }
     }


return v;
}

function doepoch() {
    var sample = [];
    var best = -1000000;
    for (var tt = 0; tt < 1000; ++tt)
    {
      g = drawRandomNormal();
      W = [];
      b = [];
      for(var k = 0; k < 4; ++k)
        W[k] = g[k];
      for(var k = 0; k < 2; ++k)
        b[k] = g[4+k];
      net = new NN(W,b);
      var total_j = 0.0;
        Q = new Quad();
        for(var i = 0; i < Points.length; ++i) {
            var j = net.jacobian(Points[i]);
            Q.insert(j.p, 1);
            total_j += j.j;
        }
        Q.fit(0.0, 3.0);
        total_j += Q.log_likelihood();
        if (total_j>best)
            best = total_j;
       sample.push({g:g,ll:total_j});
    }
    COV2 = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]];
    MEAN2 = [0,0,0,0,0,0];
    console.log(best);
    var tw=0;
    for(var tt=0;tt<100;++tt) {
        var w = Math.exp(sample[tt].ll-best); // weight
        tw += w;
        for(var k = 0; k < 6;++k)
            MEAN2[k] += w * sample[tt].g[k];
    }
    for(var k = 0; k < 6;++k) {
        MEAN2[k] /= tw;
    }
    for(var tt=0;tt<100;++tt) {
        var w = Math.exp(sample[tt].ll-best); // weight
        w /= tw;
        for(var j = 0; j < 6;++j)
             for(var k = 0; k < 6;++k)
                COV2[j][k] += w * (sample[tt].g[j] - MEAN2[j])*(sample[tt].g[k] - MEAN2[k]);
    }
    for(var j = 0; j < 6;++j) {
       MEAN[j] = 0.99*MEAN[j] + 0.01*MEAN2[j];
       for(var k = 0; k < 6;++k)
          COV[j][k] = 0.99*COV[j][k] + 0.01*COV2[j][k];
    }
	



}*/

// step #1, draw a proposal for a new NN
// step #2, transform the points and insert them into a new quad 3
// step #3, evaluate log likelihood, accept or reject, save in chain
// step #4, goto step #1

