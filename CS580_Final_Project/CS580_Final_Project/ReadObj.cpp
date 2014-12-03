#include "ReadObj.h"
using namespace std;

bool  isSpace(char token){

	return (token==' ') || (token =='\t') ;

}

float getFloat(const char *&token)
{
	float res;
		token += strspn(token, " \t");
	   res= (float)atof( token);
	    token += strcspn(token, " \t\r");
		return res;
}
int getFaceInt(const char *&token)
{
	int res;
		if(token[0]=='/' && token[1]=='/')
		{	
			token++;
				return NULL;
		}
		token += strspn(token, " /");

	   res= (  int)atoi( token);
	    token += strcspn(token, " /\r");



	return res;
}
const char* getString(const char *&token)
{

		token += strspn(token, " \t");
		int n = strcspn(token, " \t\r");
		const char *str1 = token;
		token+=n;
		return str1;
}

vector<MtlInfo> readMaterial(char *mtllibname){

	std::vector<MtlInfo> mtlList;
 
	int maxchars = 8192;  // Alloc enough size.
	std::vector<char> buf(maxchars);  // Alloc enough size.
	 
	fstream infile;
    infile.open(mtllibname,ios::in);

	while(infile.peek()!=-1){

		infile.getline(&buf[0], maxchars);

		std::string linebuf(&buf[0]);
 
		// Trim newline '\r\n' or '\n'
		if (linebuf.size() > 0) {
		  if (linebuf[linebuf.size()-1] == '\n') linebuf.erase(linebuf.size()-1);
		}
		if (linebuf.size() > 0) {
		  if (linebuf[linebuf.size()-1] == '\r') linebuf.erase(linebuf.size()-1);
		}
 
		// Skip if empty line.
		if (linebuf.empty()) {
		  continue;
		}

		const char *token = linebuf.c_str();

		if (token[0] == '\0') continue; // empty line
    
		if (token[0] == '#') continue;  // comment line

			//check new mtl
		if(strncmp(token,"newmtl",6)==0 && isSpace(token[6])){
	
			token+=7;

			MtlInfo temp;
			char mtlname[100];
 
		//	memcpy(mtlname,token,sizeof(mtlname));
			int te =mtlList.size() ;
			//.mtlname = mtlname;
			memcpy(temp.mtlname,token,sizeof(temp.mtlname));
			mtlList.push_back(temp);
			continue;
	
		}
		// ka
		if (token[0] == 'K' && token[1]=='a' && isSpace((token[2]))) {
		  token += 3;
		  float x, y, z;
	  
			x=getFloat(token);
			y=getFloat(token);
			z=getFloat(token);
 
			mtlList[mtlList.size() -1].Ka[0]= x;
			mtlList[mtlList.size() -1].Ka[1]= y;
			mtlList[mtlList.size() -1].Ka[2]= z;
		  continue;
		}
		// ks
		if (token[0] == 'K' && token[1]=='s' && isSpace((token[2]))) {
		  token += 3;
		  float x, y, z;
	  
			x=getFloat(token);
			y=getFloat(token);
			z=getFloat(token);
 
			mtlList[mtlList.size() -1].Ks[0]= x;
			mtlList[mtlList.size() -1].Ks[1]= y;
			mtlList[mtlList.size() -1].Ks[2]= z;
		  continue;
		}
		// kd
		if (token[0] == 'K' && token[1]=='d' && isSpace((token[2]))) {
		  token += 3;
		  float x, y, z;
	  
			x=getFloat(token);
			y=getFloat(token);
			z=getFloat(token);
 
			mtlList[mtlList.size() -1].Kd[0]= x;
			mtlList[mtlList.size() -1].Kd[1]= y;
			mtlList[mtlList.size() -1].Kd[2]= z;
		  continue;
		}
		// kd
		if (token[0] == 'T' && token[1]=='f' && isSpace((token[2]))) {
		  token += 3;
		  float x, y, z;
	  
			x=getFloat(token);
			y=getFloat(token);
			z=getFloat(token);
 
			mtlList[mtlList.size() -1].Tf[0]= x;
			mtlList[mtlList.size() -1].Tf[1]= y;
			mtlList[mtlList.size() -1].Tf[2]= z;
		  continue;
		}
		// Ni
		if (token[0] == 'N' && token[1]=='i' && isSpace((token[2]))) {
		  token += 3; 
			mtlList[mtlList.size() -1].Ni= getFloat(token);
		  continue;
		}
	}


	infile.close();

	return mtlList;
}
//int main(int argc, char **argv)
 
void ObjInfo::readObj(char *objName)
{
	/*
		std::vector<float> v;
		std::vector<float> vn;
		std::vector<float> vt;
		std::vector<FaceInfo> f;
		std::map<char *,vector<MtlInfo>> mltMap; 
*/
  int maxchars = 8192;  // Alloc enough size.
  std::vector<char> buf(maxchars);  // Alloc enough size.
 
  fstream infile;
    infile.open(objName,ios::in);
	
	char *glousemtl;
	char *glomtllib;
 
	MtlInfo t;
 

while(infile.peek()!=-1){

    infile.getline(&buf[0], maxchars);

    std::string linebuf(&buf[0]);
 
    // Trim newline '\r\n' or '\n'
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size()-1] == '\n') linebuf.erase(linebuf.size()-1);
    }
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size()-1] == '\r') linebuf.erase(linebuf.size()-1);
    }
 
    // Skip if empty line.
    if (linebuf.empty()) {
      continue;
    }

	const char *token = linebuf.c_str();

    if (token[0] == '\0') continue; // empty line
    
    if (token[0] == '#') continue;  // comment line

    // vertex
    if (token[0] == 'v' && isSpace((token[1]))) {
		token += 2;

		float3 temp;
		temp.x=getFloat(token)*1.5;
		temp.z=getFloat(token)*1.5;
		temp.y=getFloat(token)*1.5;
		v.push_back(temp);
      continue;
    }
	//normal
    if (token[0] == 'v' &&token[1] == 'n' && isSpace((token[2]))) {
		token += 3;
	  
		float3 temp;
		temp.x=getFloat(token);
		temp.z=getFloat(token);
		temp.y=getFloat(token);
		vn.push_back(temp);
      continue;
    }

	//uv
    if (token[0] == 'v' &&token[1] == 't' && isSpace((token[2]))) {
      token += 3;
		float2 temp;
		temp.x=getFloat(token);
		temp.y=getFloat(token);
		vt.push_back(temp);
      continue;
    }
	//get mtl lib
	if(strncmp(token,"mtllib",6)==0 && isSpace(token[6])){
	
		token+=7;
		char mtllibbuf[100];
		
		memcpy(mtllibbuf,token,100);
		glomtllib = mtllibbuf;
 
 
		mltMap.insert(pair<char *,std::vector<MtlInfo>>(mtllibbuf,readMaterial(mtllibbuf))) ;
		//std::cout << "a => " << mltMap.find("mtllibbuf")->second<<'\n';
 
 
		map<char *,std::vector<MtlInfo>>::iterator it;
		it  =  mltMap.find(mtllibbuf ) ;
		//vector<MtlInfo> ma((it->second).begin(),it->second.end());
		vector<MtlInfo> ma;
		ma =it->second;
		cout<<ma[0].mtlname;
	      continue;
	
	}
	//get mtl name
	if(strncmp(token,"usemtl",6)==0 && isSpace(token[6])){
	
		token+=7;
		char namebuf[100];
		//glousemtl = 
		memcpy(namebuf,token,100);
		glousemtl = namebuf;
	      continue;
	
	}

	// face info
    if (token[0] == 'f' &&  isSpace(token[1])) {
	
		token+=2;
		FaceInfo tempf;
		
		const char *a = getString(token);
		const	 char *b = getString(token);
		const char *c = getString(token);
		
		const char *strlist[3] = {a,b,c}; 
		for(int c=0;c<3;c++)
		{
			tempf.vertexIndex[c] = getFaceInt(strlist[c]);
			tempf.uvIndex[c] = getFaceInt(strlist[c]);
			tempf.normalIndex[c] = getFaceInt(strlist[c]);	
		}
		tempf.usemtl = glousemtl;
		f.push_back(tempf);
		continue;
	}
	
  }// end while

 infile.close();
 //   return 0;
}

