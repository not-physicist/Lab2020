#include "TROOT.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH2.h"

//
//
// 
//
//

TString ee_cuts = "Pcharged < 200 && cos_thet <= 1 && Ncharged < 4 && E_ecal > 60 && E_hcal < 2 && cos_thet > -0.9 && cos_thet < 0.5";

TString mm_cuts = "Pcharged < 200 && cos_thet <=1 && Ncharged < 4 && E_ecal < 60 && Pcharged > 70";

TString tt_cuts = "Pcharged < 200 && Ncharged < 5 && Pcharged < 60 && Pcharged > 1 && E_ecal < 70";

TString qq_cuts = "Pcharged < 200 && Ncharged > 10";

TString s1 = "E_lep > 44.0 && E_lep < 44.5 ";
TString s2 = "E_lep > 44.5 && E_lep < 45 ";
TString s3 = "E_lep > 45 && E_lep < 45.2 ";
TString s4 = "E_lep > 45.2 && E_lep < 45.7 ";
TString s5 = "E_lep > 45.7 && E_lep < 46.2 ";
TString s6 = "E_lep > 46.2 && E_lep < 46.7 ";
TString s7 = "E_lep > 46.7 && E_lep < 48.0 ";

TString fd = "cos_thet >= 0 && cos_thet <= 1 ";
TString bd = "cos_thet >= -1 && cos_thet <= 0 ";

void cutplotter(TString FileName="ERROR", TString BranchName="ERROR",TString cuts="")
{
	 if(FileName=="ERROR" && BranchName!="ERROR"){
		cout << "***ERROR: Did you forget to specify the file name?"  << endl;
		 return ;
	 }
	
	 else if(FileName!="ERROR" && BranchName=="ERROR"){
		cout << "***ERROR: Did you forget to specify the branch name?"  << endl;
		 return ;
	 }
	 else if(FileName=="ERROR" && BranchName=="ERROR"){
		cout << "***ERROR: Did you forget to specify both the file and branch names?"  << endl;
		 return ;
	 }
	else{
	TString ttreename;
        if(FileName=="data1" || FileName=="data2" || FileName=="data3"|| FileName=="data4"|| FileName=="data5"||FileName=="data6") ttreename="h33";
         else ttreename="h3";
	if(cuts=="") cout << "***CUTS: You have not applied any cuts. Plot will be drawn without cuts."  << endl;	
	else cout << "***CUTS: Following cut(s) will be applied to the plot: "+cuts  << endl;	
         TFile *f = new TFile(FileName+".root");
         TTree *t1 = (TTree*)f->Get(ttreename);
         //else TTree *t1 = (TTree*)f->Get("h3");
	 TCanvas *c1 = new TCanvas("c1", "Process:"+FileName+" | Variable: "+BranchName+" | Selection: "+cuts,200,10,900,700);

         float E_ecal, E_hcal,E_lep;
         float Ncharged, Pcharged;
         float cos_thet, cos_thru;

         t1->SetBranchAddress("Ncharged",&Ncharged);
         t1->SetBranchAddress("Pcharged",&Pcharged);
         t1->SetBranchAddress("E_ecal",&E_ecal);
         t1->SetBranchAddress("E_hcal",&E_hcal);
         t1->SetBranchAddress("E_lep",&E_lep);
         t1->SetBranchAddress("cos_thru",&cos_thru);
         t1->SetBranchAddress("cos_thet",&cos_thet);
	 c1->cd();
         t1->Draw(BranchName,cuts);

         TH1F *hist = (TH1F*)gPad->GetPrimitive("htemp");
         hist->SetTitle("Process:"+FileName+" | Variable:"+BranchName+" | Cut:"+cuts);
         c1->Update();
         c1->Modified();
	}
}
