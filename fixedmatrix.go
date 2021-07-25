package main
import (
	"strconv"
	"fmt"
	"math"
	"bytes"
	"errors"
	"encoding/gob"
	"io"
)

var (
	ErrShape = errors.New("mat: dimension mismatch")
)

type fixed int64

const maxLen = int64(int(^uint(0) >> 1))

type Matrix struct {
	row, col int;
	data [][]fixed;
}

func NewMatrix(r, c int, nums []fixed) *Matrix {
	data := make([][]fixed, r);
	for i:=0; i<r; i++{
		data[i] = make([]fixed, c);
		if nums != nil {
			for j:=0; j<c; j++{
				data[i][j] = nums[i*c + j];
			}
		}
	}
	mat := Matrix{row: r, col: c, data: data};
	return &mat;
}

func Copy(m *Matrix) *Matrix {
	data := make([]fixed, m.row*m.col)
	for i:=0; i<m.row; i++{
		for j:=0; j<m.col; j++{
			data[i*m.col + j] = m.data[i][j]
		}
	}
	return NewMatrix(m.row, m.col, data)
}

func (m *Matrix) Dims() (r,c int){
	return m.row, m.col
}

func (m *Matrix) Set(r, c int, val fixed){
	m.data[r][c] = val
}

func (m *Matrix) At(r, c int) fixed {
	return m.data[r][c]
}

func (m *Matrix) MulElem(a, b *Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			foo := MultiplyFixed(a.At(r, c),b.At(r, c))
			m.Set(r, c, foo)
		}
	}
}

func (m *Matrix) Add(a, b *Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			foo := a.At(r, c)+b.At(r, c)
			m.Set(r, c, foo)
		}
	}
}

func (m *Matrix) Sub(a, b *Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			foo := a.At(r, c)-b.At(r, c)
			m.Set(r, c, foo)
		}
	}
}
// As long as a or b is not m, this works fine
func (m *Matrix) Product(a, b *Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(ErrShape)
	}

	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			var sum fixed = 0
			for k := 0; k < ac; k++ {
				sum += MultiplyFixed(a.At(i,k),b.At(k,j))
			}
			m.Set(i, j, sum)
		}
	}
}

func (m *Matrix) T() *Matrix{
	var newCol, newRow int = m.Dims();
	newData := make([]fixed, newRow*newCol);
	for i:=0; i<newRow; i++ {
		for j:=0; j<newCol; j++{
			newData[i*newCol + j] = m.data[j][i];
		}
	}
	return NewMatrix(newRow,newCol,newData);
}

func (m *Matrix) Scale(c fixed){
	r,col := m.Dims();
	for i:=0; i<r; i++{
		for j:=0; j<col; j++{
			foo := MultiplyFixed(m.At(i,j),c)
			m.Set(i,j,foo);
		}
	}
}

func (m *Matrix) Apply(fn func(i, j int, v fixed) fixed, a *Matrix){
	ar, ac := a.Dims();
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.Set(r, c, fn(r, c, a.At(r, c)))
		}
	}
}

func Min(m *Matrix) fixed{
	min := m.At(0,0)
	for r:=0;r<m.row;r++{
		for c:=0;c<m.col;c++{
			if(m.At(r,c)<min){
				min = m.At(r,c)
			}
		}
	}
	return min
}

func Max(m *Matrix) fixed{
	max := m.At(0,0)
	for r:=0;r<m.row;r++{
		for c:=0;c<m.col;c++{
			if(m.At(r,c)>max){
				max = m.At(r,c)
			}
		}
	}
	return max
}

//need to account for things like negative numbers properly, fix after i optimize this to not use loops;
func MultiplyFixed(a, b fixed) fixed{
    var isNegative bool = false;
    if(a < 0){
	    a = -a
	    isNegative = !isNegative
  	}
    if(b < 0){
    	b = -b
    	isNegative = !isNegative
  	}
  //((A+B) * (C+D) = (A*C) + (C*B) + (A*D) + (B*D)

  //A*C
  A := a >> 48
  C := b >> 48
  res := (A * C) << 48

  B := a & 0xffffffffffff
  D := b & 0xffffffffffff

  //C*B
  res += (C * B)

  //A*D
  res += (A * D)

  //B*D
  res += ((B>>24) * (D>>24))

  if((isNegative && res >= 0) || (!isNegative && res < 0)) {
  	res = -res
  }
  return res
}

func DivideFixed(a, b fixed) fixed{
  var isNegative bool = false;
  if(a < 0){
    a = -a
    isNegative = !isNegative
	}
  if(b < 0){
    b = -b
    isNegative = !isNegative
  }
  if(b == 0){
  	fmt.Printf("Divide By Zero Error\n")
    return -fixed(0x7fffffffffffffff)
  }
  if(b == (b>>48)<<48){
    return a / (b>>48)
  }
  res := (a / (b >> 24)) << 24
  if((isNegative && res >= 0) || (!isNegative && res < 0)) {
    res = -res
  }
  return res
}

func fixedMax(a, b fixed) fixed{
	if(a >= b){
		return a
	}
	return b
}

func fixedMin(a, b fixed) fixed{
	if(a <= b){
		return a
	}
	return b
}

func abs(x int64)int64{
  if x < 0{
    return -x
  }
  return x
}

func scale_2(x fixed, n int64)fixed{
  var i int64
  for i = 0; i < abs(n); i++ {
      if(n < 0) {
        x /= 2
      } else {
        x *= 2
      }
  }
  return x
}

func floor(x fixed)fixed{
	xp := fixed(uint64(x) & 0xffff000000000000)
	if(xp == x){
		return x
	}
	if(x > 0){
		return xp
	} else {
		return xp - 0x1000000000000
	}
}

func ceil(x fixed)fixed{
	xp := fixed(uint64(x) & 0xffff000000000000)
	if(xp == x){
		return x
	}
	if(x < 0){
		return xp
	} else {
		return xp + 0x1000000000000
	}
}

func round(x fixed)fixed{
	if(fixed(uint64(x) & 0xffff800000000000) <= x){
		return ceil(x)
	} else {
		return floor(x)
	}
}

func bit(x fixed, n int) bool {
   return (x & (1 << (n-1))) != 0
}

const LN2 fixed = 0xB17217F7D1CF
const LN2_H fixed = 0xB17217F7D1CE
const LN2_L fixed = 0x1

const INV_LN2 fixed = 0x171547652B82F

const ONE_HALF fixed = 0x800000000000
const ONE fixed = 0x1000000000000
const TWO fixed = 0x2000000000000

const P1 fixed = 0x2AAAAAAAAAAA
const P2 fixed = -0x00B60B60B60B
const P3 fixed = 0x0004559AAF00
const P4 fixed = -0x00001BBD0000
const P5 fixed = 0x000000B20000

func exp(x fixed)fixed{
  var hi fixed
  var lo fixed
  var k int64
  var t fixed = fixed(abs(int64(x)));
  if(t > LN2 / 2){ //if abs(x) > ln(2)/2
    if(t < MultiplyFixed(ONE_HALF + ONE, LN2)){
      hi = t - LN2_H
      lo = LN2_L
      k = 1
    } else {
      k = toInt((MultiplyFixed(INV_LN2, t) + ONE_HALF))
      k_fixed := fixed(k << 48)
      hi = t - MultiplyFixed(k_fixed, LN2_H)
      lo = MultiplyFixed(k_fixed, LN2_L)
    }
    if(x < 0){
      hi = -hi
      lo = -lo
      k = -k
    }
    x = hi - lo
  } else if(t < 0x10){ //if x is close to 0
    return 0x1 << 48
  } else{
    lo = 0
    hi = 0
    k = 0
  }
  //now x is in primary range.
  t = MultiplyFixed(x, x)
  P4_5 := P4 + MultiplyFixed(t, P5)
  P3_5 := P3 + MultiplyFixed(t, P4_5)
  P2_5 := P2 + MultiplyFixed(t, P3_5)
  P1_5 := P1 + MultiplyFixed(t, P2_5)
  c := x - MultiplyFixed(t, P1_5)
  if k == 0 {
    return ONE - (lo - DivideFixed(MultiplyFixed(x , c), TWO - c) - x)
  }
  y := ONE - (lo - DivideFixed(MultiplyFixed(x , c), TWO - c) - hi) 
  return scale_2(y, k)
}

/*func exp_test(x fixed)fixed{
	P5_5 := floatToFixed(1.0/720) + MultiplyFixed(x, floatToFixed(1.0/5040))
  P4_5 := floatToFixed(1.0/120) + MultiplyFixed(x, P5_5)
  P3_5 := floatToFixed(1.0/24) + MultiplyFixed(x, P4_5)
  P2_5 := floatToFixed(1.0/6) + MultiplyFixed(x, P3_5)
  P1_5 := floatToFixed(1.0/2) + MultiplyFixed(x, P2_5)
  return ONE + x + MultiplyFixed(x, P1_5)
}

func ln(x fixed)fixed{ //TODO
 return fixed(0) 
}

func pow(base, exponent fixed)fixed{ //TODO
	return exp(MultiplyFixed(ln(base), exponent))
}

func log(base, val fixed)fixed{ //TODO
  return DivideFixed(ln(val), ln(base))
}*/

func toInt(x fixed)int64{
  isNegative := x < 0
  x = x >> 48
  if(isNegative){
    x = -x 
  }
  return int64(x)
}

func intToFixed(x int) fixed{
	isNegative := x < 0
  x = x << 48
  if(isNegative){
    x = -x 
  }
	return fixed(x);
}

func floatToFixed(a float64) fixed{
	isNegative := a < 0
	if(isNegative){
		a = -a
	}
	i := int64(math.Float64bits(a))
	mantissa := (i & 0xfffffffffffff)+0x10000000000000
	exponent := (i>>52)-1023
	output := mantissa>>4
	if(exponent < 0){
		exponent = -exponent
		output = output>>exponent
	} else {
		output = output<<exponent
	}
	if(isNegative){
		output = -output
	}
	return fixed(output)
}

func toFloat(a fixed) float64{
	var out float64
	out = 1.0
	return (out * float64(int64(a)))/0x1000000000000
}

func (f fixed) String() string {
	s := ""
	if(f<0){
		s += "-"
		f = -f
	}
	b := f & 0xffffffffffff
	s += fmt.Sprintf("%d", f >> 48)
	s += strconv.FormatFloat(toFloat(b), 'f', -1, 64)[1:]
  return s 
}

func (m *Matrix) Resize(r, c int){
	data := make([][]fixed, r);
	for i:=0; i<r; i++{
		data[i] = make([]fixed, c);
		for j:=0; j<c; j++{
			if((i<m.row) && (j < m.col)){
				data[i][j] = m.data[i][j];
			} else {
				data[i][j] = fixed(0)
			}
		}
	}
	m.data = data
}

type wrapMatrix struct {
	Row, Col int;
	Data [][]fixed;
}

func (m *Matrix) MarshalBinaryTo() (io.Reader, error) {
  w := wrapMatrix{m.row, m.col, m.data}
  var buf bytes.Buffer
  enc := gob.NewEncoder(&buf)
  if err := enc.Encode(w); err != nil {
    return nil, err
  }
  return bytes.NewReader(buf.Bytes()), nil
}

func (m *Matrix) UnmarshalBinaryFrom(reader io.Reader) error {
  w := wrapMatrix{}
  dec := gob.NewDecoder(reader)
  if err := dec.Decode(&w); err != nil {
    return err
  }
  m.row = w.Row
  m.col = w.Col
  m.data = w.Data
  return nil
}