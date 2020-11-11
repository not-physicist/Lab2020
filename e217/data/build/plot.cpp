/* ROOT includes */
#include <TH1F.h>
#include <TH2F.h>
#include <TClonesArray.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <iostream>

void plot(){	
	TCanvas *c1 = new TCanvas("c1","c1",1000, 500);
	gStyle->SetOptStat(kFALSE);
	
	TFile *f = new TFile("recoAna.root","READ");
	
	// BB
	TH1F *h_BB = (TH1F *)f->Get("hits_BB");
	h_BB->SetLineColor(kRed);
	h_BB->SetTitle("Number of hits per straw for each layer in bottom module");
	h_BB->GetYaxis()->SetRangeUser(0, 1200);
	h_BB->Draw();
	
	// BM
	TH1F *h_BM = (TH1F *)f->Get("hits_BM");
	h_BM->SetLineColor(kBlue);
	h_BM->Draw("same");

	// BT
	TH1F *h_BT = (TH1F *)f->Get("hits_BT");
	h_BT->SetLineColor(kOrange);
	h_BT->Draw("same");
	
	TLegend *legend = new TLegend(0.1,0.7,0.2,0.5);
	legend->AddEntry(h_BB, "bottom");
	legend->AddEntry(h_BM, "middle");
	legend->AddEntry(h_BT, "top");
	
	legend->Draw();
	c1->SaveAs("hitsLayers.pdf");
	
	// BTot
	TH1F *h_BTot = (TH1F *)f->Get("hits_BTot");
	h_BTot->SetTitle("Number of hits per straw for all layers in each module");
	h_BTot->GetYaxis()->SetRangeUser(0, 4000);
	h_BTot->SetLineColor(kRed);
	h_BTot->Draw();

	TH1F *h_TTot = (TH1F *)f->Get("hits_TTot");
	h_TTot->SetLineColor(kBlue);
	h_TTot->Draw("same");
	
	TLegend *legend2 = new TLegend(0.1,0.7,0.2,0.5);
	legend2->AddEntry(h_BTot, "bottom");
	legend2->AddEntry(h_TTot, "top");

	legend2->Draw();
	c1->SaveAs("hitsModules.pdf");

	// AngDistri
	TH1F *h_ang = (TH1F *)f->Get("AngDistri");
	h_ang->Draw();
	
	// fit
	TF1 *func = new TF1("fit", "[0]*cos(x)^[1]", -0.5, 0.5);
	func->SetParameters(0, 500);
	func->SetParameters(1, 2);
	//func->Draw("same");
	func->SetParNames("prop. const.", "power");
	h_ang->Fit("fit", "R");

	TLegend *legend3 = new TLegend(0.1,0.8,0.2,0.9);
	legend3->AddEntry(func, "[0]*cos(x)^[1]");
	legend3->AddEntry(h_ang, "Angle");	
	legend3->Draw();

	c1->SaveAs("angDistri.pdf");
}
