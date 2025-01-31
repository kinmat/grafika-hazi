//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Matok Kinga
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 20;


//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 10;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	void Animate(float t) { }
};

//---------------------------
template<class T> struct Dnum {
	//---------------------------
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum<T> operator+(Dnum<T> r) { return Dnum<T>(f + r.f, d + r.d); }
	Dnum<T> operator-(Dnum<T> r) { return Dnum<T>(f - r.f, d - r.d); }
	Dnum<T> operator*(Dnum<T> r) { return Dnum<T>(f * r.f, f * r.d + d * r.f); }
	Dnum<T> operator/(Dnum<T> r) {
		float l = r.f * r.f;
		return (*this) * Dnum<T>(r.f / l, -r.d / l);
	}
};

template<class T>  Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sin(g.f), cos(g.f) * g.d); }
template<class T>  Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cos(g.f), -sin(g.f) * g.d); }
template<class T>  Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T>  Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), 1 / g.f * g.d); }
template<class T>  Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T>  Dnum<T> Pow(Dnum<T> g, float n) { return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }
template<class T>  Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T>  Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T>  Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }

typedef Dnum<vec2> Dnum2;

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float t) {	}
};

class CsikosTexture : public Texture {
	//---------------------------
public:
	CsikosTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), black(0, 0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] =  (x & 1) ? yellow : black;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class AtmenetesTexture : public Texture {
	//---------------------------
public:
	AtmenetesTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = vec4(1-x*0.03f, 0, x*0.1f, 1);
		}
		create(width, height, image, GL_NEAREST);
	}
};

class TetraTexture : public Texture {
	//---------------------------
public:
	TetraTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = vec4(0.8f, 0, 0.7f, 1);
		}
		create(width, height, image, GL_NEAREST);
	}
};


//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};


//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

std::vector<VertexData> nyulvanyok;

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	void virtual eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vd.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vd.normal = cross(drdU, drdV);
		return vd;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				VertexData egy = GenVertexData((float)j / M, (float)i / N);
				VertexData ketto = GenVertexData((float)j / M, (float)(i + 1) / N);
				vtxData.push_back(egy);
				vtxData.push_back(ketto);
				if(i%2==0&&j%2==0) nyulvanyok.push_back(egy);
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class Tetraeder : public Geometry {
	unsigned int nVtx;
public:
	Tetraeder() {
		nVtx = 0;
		create();
	}

	vec3 triangleNorm(vec3 a, vec3 b, vec3 c) {
		vec3 ab = b - a;
		vec3 ac = c - a;
		return cross(ab, ac);
	}

	std::vector<VertexData> triVTX(vec3 a, vec3 b, vec3 c, int depth) {
		std::vector<VertexData> vtx;
		VertexData egy, ketto, harom;
		egy.position = a;
		ketto.position = b;
		harom.position = c;
		vec3 norm = triangleNorm(a, b, c);
		egy.normal = ketto.normal = harom.normal = norm;
		egy.texcoord = ketto.texcoord = harom.texcoord = vec2(0, 0);
		vtx.push_back(egy);
		vtx.push_back(ketto);
		vtx.push_back(harom);



		std::vector<vec3> felezo1 = felezoPontok(a, b, c, depth);
		std::vector<VertexData> felezoTetra = tetraVTX(felezo1[0], felezo1[1], felezo1[2], felezo1[3],depth+1);
		for (int i = 0; i < felezoTetra.size(); i++) vtx.push_back(felezoTetra[i]);

		return vtx;
	}

	std::vector<VertexData> ujraOldalFelez(vec3 c, vec3 b, vec3 a, int depth) {
		std::vector<VertexData> vtx;
		std::vector<vec3> felezo = felezoPontok(c, b, a, depth);
		std::vector<vec3> felezo2 = felezoPontok(b, felezo[1], felezo[0], depth + 2);
		std::vector<VertexData> oldalTetra = tetraVTX(felezo2[0], felezo2[1], felezo2[2], felezo2[3], depth + 2);
		for (int i = 0; i < oldalTetra.size(); i++) vtx.push_back(oldalTetra[i]);

		felezo2 = felezoPontok(c, felezo[0], felezo[2], depth + 2);
		oldalTetra = tetraVTX(felezo2[0], felezo2[1], felezo2[2], felezo2[3], depth + 2);
		for (int i = 0; i < oldalTetra.size(); i++) vtx.push_back(oldalTetra[i]);

		felezo2 = felezoPontok(a, felezo[2], felezo[1], depth + 2);
		oldalTetra = tetraVTX(felezo2[0], felezo2[1], felezo2[2], felezo2[3], depth + 2);
		for (int i = 0; i < oldalTetra.size(); i++) vtx.push_back(oldalTetra[i]);

		return vtx;
	}


	std::vector<VertexData> tetraVTX(vec3 a, vec3 b, vec3 c, vec3 d, int depth) {
		std::vector<VertexData> vtx;
		std::vector<VertexData> oldalTetra;

		if (depth >2) return vtx;

	
		std::vector<VertexData> tri1 = triVTX(c, d,b, depth);
		for (int i = 0; i < tri1.size(); i++) vtx.push_back(tri1[i]);
		oldalTetra = ujraOldalFelez(c, d, b, depth);
		for (int i = 0; i < oldalTetra.size(); i++) vtx.push_back(oldalTetra[i]);

		std::vector<VertexData> tri2 = triVTX(c, a, d, depth);
		for (int i = 0; i < tri2.size(); i++) vtx.push_back(tri2[i]);
		oldalTetra = ujraOldalFelez(c, a, d, depth);
		for (int i = 0; i < oldalTetra.size(); i++) vtx.push_back(oldalTetra[i]);

		std::vector<VertexData> tri3 = triVTX(c, b, a, depth);
		for (int i = 0; i < tri3.size(); i++) vtx.push_back(tri3[i]);
		oldalTetra = ujraOldalFelez(c, b, a, depth);
		for (int i = 0; i < oldalTetra.size(); i++) vtx.push_back(oldalTetra[i]);

		std::vector<VertexData> tri4 = triVTX(a, b, d, depth);
		for (int i = 0; i < tri4.size(); i++) vtx.push_back(tri4[i]);
		oldalTetra = ujraOldalFelez(a, b, d, depth);
		for (int i = 0; i < oldalTetra.size(); i++) vtx.push_back(oldalTetra[i]);

		return vtx;
	}

	std::vector<vec3> felezoPontok(vec3 a, vec3 b, vec3 c, int depth) {
		std::vector<vec3> pont;
		vec3 ab = vec3((a + b) / 2);
		vec3 bc = vec3((c + b) / 2);
		vec3 ac = vec3((a + c) / 2);
		pont.push_back(ab);
		pont.push_back(bc);
		pont.push_back(ac);
		vec3 norm=normalize(triangleNorm(a, b, c));
		if (depth > 0) norm = norm / (depth*2);
		vec3 sulypont = (a + b + c) / 3;
		vec3 d =sulypont + norm;
		pont.push_back(d);
		return pont;
	}

	void create() {
		std::vector<VertexData> vtxData;	// vertices on the CPU
		vec3 a = vec3(1.2f, 0, 0);
		vec3 b = vec3(-1.2f, 0, 0);
		vec3 c = vec3(0, 1.5f, -0.75f);
		vec3 d = vec3(0, 0, -1.5f);

		vtxData = tetraVTX(a, b, c, d,0);


		nVtx = vtxData.size();
		glBufferData(GL_ARRAY_BUFFER, nVtx * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
};


//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2 * (float)M_PI;
		V = V * M_PI;
		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);

	}
};

class Tractricoid : public ParamSurface {
	//---------------------------
public:
	Tractricoid() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float height = 3.0f;
		U = U * height, V = V * 2 * M_PI;
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);

	}

};


//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;

public:
	Object() {}
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend;
	}
};

struct Nyulvany : public Object {
	mat4 trac;
public:
	Nyulvany(Shader* _shader, Material* _material, Texture* _texture){
		scale = vec3(1, 1, 1);
		translation=vec3(0, 0, 0); 
		rotationAxis=vec3(-1, 0, 0);
		rotationAngle=17;
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = new Tractricoid();
	}

	void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) *TranslateMatrix(translation)* trac;
		Minv = TranslateMatrix(-translation)* trac* ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend;
	}

};

float rnd() { return (float)rand() / RAND_MAX; }

//https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector
mat4 TracMatrix(VertexData pont) {
	vec3 norm = normalize(pont.normal);
	float hossz = sqrt(norm.x * norm.x + norm.y * norm.y);
	return mat4(vec4((norm.y/hossz), (-norm.x/hossz), 0, 0),
				vec4((norm.x*norm.z/hossz), (norm.y * norm.z / hossz), -hossz, 0),
				vec4(norm.x, norm.y, norm.z, 0),
				vec4(0, 0, 0, 1));
}
//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	Camera camera; // 3D camera
	std::vector<Light> lights;
public:
	void Build() {
		// Shaders
		Shader* phongShader = new PhongShader();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Texture* csikos = new CsikosTexture(40, 40);
		Texture* atmenet = new AtmenetesTexture(30, 30);
		Texture* sima = new TetraTexture(5, 5);

		// Geometries
		Geometry* sphere = new Sphere();
		Geometry* tetra = new Tetraeder();

		// Create objects by setting up their vertex data on the GPU
		Object* sphereObject1 = new Object(phongShader, material0, csikos, sphere);
		sphereObject1->translation = vec3(0, 0, -1);
		sphereObject1->rotationAxis = vec3(0, 1, 1);
		sphereObject1->scale = vec3(1, 1, 1);
		objects.push_back(sphereObject1);

		Object* harom = new Object(phongShader, material0, sima, tetra);
		harom->scale = vec3(1,1, 1);
		harom->translation = vec3(2, 2, 0);
		harom->rotationAxis = vec3(1, 1, 1);
		objects.push_back(harom);

		int db = nyulvanyok.size();
		for (int i = 0; i < db; i++) {
			if (i % 2== 0) {
				Nyulvany* tracObj = new Nyulvany(phongShader, material0, atmenet);
				tracObj->translation = nyulvanyok[i].position;
				tracObj->trac = TracMatrix(nyulvanyok[i]);
				tracObj->scale = vec3(0.2f, 0.2f, 0.2f);
				objects.push_back(tracObj);
			}

		}
		
		
		// Camera
		camera.wEye = vec3(0, 0, 6);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
		lights[0].La = vec3(1, 1, 1);
		lights[0].Le = vec3(3, 3, 3);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		camera.Animate(tend);
		for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		for (Object* obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is �infinitesimal�
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}