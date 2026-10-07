import tkinter as tk
from tkinter import ttk, messagebox
from model import CreditModel,FIELDS,NUMERIC,LABELS

class CreditApp:
    def __init__(self,root):
        self.root=root;self.model=CreditModel();self.current_id=None
        root.title('Credit Scoring Demo');root.geometry('900x740');root.minsize(650,500)
        root.configure(bg='#F4F2F8');root.columnconfigure(0,weight=1);root.rowconfigure(3,weight=1)
        header=tk.Frame(root,bg='#4C356A',padx=18,pady=15);header.grid(row=0,column=0,sticky='ew')
        tk.Label(header,text='Credit Scoring Model',font=('Segoe UI',22,'bold'),fg='white',bg='#4C356A').pack(anchor='w')
        tk.Label(header,text='German Credit dataset | Logistic Regression',fg='white',bg='#4C356A').pack(anchor='w')
        tk.Label(root,text='Historical dataset demo. Scores are not a bank credit rating or a lending decision.',bg='#F4F2F8',wraplength=650).grid(row=1,column=0,pady=8)
        controls=tk.Frame(root,bg='#F4F2F8');controls.grid(row=2,column=0,sticky='ew',padx=16)
        tk.Label(controls,text='Test sample:',bg='#F4F2F8').pack(side='left')
        self.selector=ttk.Combobox(controls,state='readonly',width=16,values=[f'Sample {i+1}' for i in range(200)]);self.selector.pack(side='left',padx=8);self.selector.current(0)
        self.selector.bind('<<ComboboxSelected>>',lambda e:self.load_sample())
        ttk.Button(controls,text='Load sample',command=self.load_sample).pack(side='left',padx=4)
        ttk.Button(controls,text='Clear',command=self.clear).pack(side='left',padx=4)
        ttk.Button(controls,text='Evaluation',command=self.evaluation).pack(side='right')
        container=tk.Frame(root);container.grid(row=3,column=0,sticky='nsew',padx=16,pady=10);container.columnconfigure(0,weight=1);container.rowconfigure(0,weight=1)
        canvas=tk.Canvas(container,bg='white',highlightthickness=0);canvas.grid(row=0,column=0,sticky='nsew')
        scroll=ttk.Scrollbar(container,orient='vertical',command=canvas.yview);scroll.grid(row=0,column=1,sticky='ns');canvas.configure(yscrollcommand=scroll.set)
        form=tk.Frame(canvas,bg='white');wid=canvas.create_window((0,0),window=form,anchor='nw');form.columnconfigure(1,weight=1)
        form.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(wid,width=e.width))
        self.inputs={}
        for row,name in enumerate(FIELDS):
            tk.Label(form,text=LABELS[name],bg='white',anchor='w').grid(row=row,column=0,sticky='w',padx=12,pady=9)
            var=tk.StringVar();self.inputs[name]=var
            if name in NUMERIC: widget=ttk.Entry(form,textvariable=var)
            else: widget=ttk.Combobox(form,textvariable=var,state='readonly',values=self.model.options[name])
            widget.grid(row=row,column=1,sticky='ew',padx=12,pady=9)
            var.trace_add('write',self.changed)
        self.result=tk.StringVar()
        self.predict_button=tk.Button(root,text='Predict credit risk',command=self.predict,bg='#4C356A',fg='white',font=('Segoe UI',12,'bold'),pady=10)
        self.predict_button.grid(row=4,column=0,sticky='ew',padx=16,pady=5)
        tk.Label(root,textvariable=self.result,bg='#F4F2F8',justify='left',anchor='w',wraplength=800,font=('Segoe UI',11)).grid(row=5,column=0,sticky='ew',padx=16,pady=(5,12))
        root.bind('<Return>',self.predict);self.load_sample()

    def changed(self,*args):
        self.current_id=None
        if hasattr(self,'result'):self.result.set('Values changed. Click Predict to classify.')
    def load_sample(self):
        pos=self.selector.current()
        if pos<0:return
        index=int(self.model.test_ids[pos])
        for name,var in self.inputs.items():var.set(str(self.model.frame.iloc[index][name]))
        self.current_id=index
        self.result.set(f'Sample {pos+1} loaded. Click Predict credit risk. Category codes are explained in data/german.doc.')
    def predict(self,event=None):
        try:
            label,score=self.model.predict({name:var.get() for name,var in self.inputs.items()})
            text=f'Predicted dataset class: {label.upper()} credit risk\nModel score for bad credit risk: {score:.1%}'
            if self.current_id is not None:
                actual='bad' if self.model.target.iloc[self.current_id] else 'good'
                text+=f'\nRecorded label: {actual.upper()} | '+('Match' if actual==label else 'Mismatch')
            self.result.set(text)
        except ValueError as error:self.result.set('Input error: '+str(error))
        except Exception as error:
            print('Prediction failed:',error);self.result.set('Prediction failed. Check the terminal for details.')
    def clear(self):
        for var in self.inputs.values():var.set('')
        self.current_id=None;self.result.set('Load a sample or fill every field.')
    def evaluation(self):
        m=self.model.metrics
        text=f'Training: 800 | Test: 200\nPositive class: bad credit risk\n\n'
        for name in ['accuracy','precision_bad','recall_bad','f1_bad','roc_auc_bad','majority_baseline_accuracy']:text+=f'{name.replace("_"," ").title()}: {m[name]:.4f}\n'
        text+='\nConfusion matrix (actual rows, predicted columns; good, bad):\n'+str(m['confusion_matrix'])
        messagebox.showinfo('Held-out evaluation',text,parent=self.root)

if __name__=='__main__':
    root=tk.Tk();CreditApp(root);root.mainloop()
