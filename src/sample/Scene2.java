package sample;
import com.sun.javafx.application.HostServicesDelegate;
import com.sun.javafx.css.StyleManager;
import javafx.application.Application;
import javafx.application.HostServices;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;

import java.awt.*;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;

import java.util.*;
import java.io.*;

public class Scene2 {
    public ScrollPane scrollpane;


    @FXML
    Label nameLabel,nameLabel1;
    private Stage stage;
    private Scene scene;
    private Parent root;
    public String result="";
    @FXML
    private CheckBox mBox,fBox;
    @FXML
    ImageView product1img, product2img, product3img;



    public void displayOutput(String username) {


        try{
            FileWriter fw=new FileWriter("name.txt");
            fw.write(username);
            fw.close();
        }

        catch(Exception e){
            System.out.println(e);
        }
        try{
            List<String> commands = new ArrayList<>();
            commands.add("python.exe "); // command
            commands.add("model.py");
            ProcessBuilder pb = new ProcessBuilder(commands);
            Process p = pb.start();
            p.waitFor();
        }catch(Exception e){
            e.printStackTrace();
        }

        try{
            File file = new File("results.txt");
            Scanner sc;
            sc = new Scanner(file);
            while (sc.hasNextLine()){
                result += sc.nextLine();
            }
            sc.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }





        mBox.setDisable(true);
        fBox.setDisable(true);
        mBox.setStyle("-fx-opacity: 1");
        fBox.setStyle("-fx-opacity: 1");

        fBox.setSelected(false);
        mBox.setSelected(false);

        if (result.equals("Female")) {
            fBox.setSelected(true);
        }
        else{
            mBox.setSelected(true);
        }
        nameLabel.setText(username + " is most likely a " + result + " name");
        nameLabel1.setText(username+" would possibly like these products :");
        Random rand = new Random();

        // Generate random integers in range 0 to 9
        int rand_int1 = rand.nextInt(10);
        int rand_int2 = rand.nextInt(10);
        int rand_int3 = rand.nextInt(10);

        while (true){
            if(rand_int1 == rand_int2 | rand_int2 ==rand_int3 ){
                rand_int2 = rand.nextInt(10);
                rand_int3 = rand.nextInt(10);
                continue;
            }
           break;
        }

        displayim(result,rand_int1,rand_int2,rand_int3);

    }

    public void displayim(String gender, int n1, int n2,int n3){
//        System.out.println(gender);
        String[] fmprod = {"f1.png","f2.png","f3.png","f4.png","f5.png","f6.png","f7.png","f8.png","f9.png","f10.png"};
        String[] mprod = {"m1.png","m2.png","m3.png","m4.png","m5.png","m6.png","m7.png","m8.png","m9.png","m10.png"};
        if(gender.equals("Female")) {
            Image newimg1 = new Image(this.getClass().getResourceAsStream(fmprod[n1]));
            Image newimg2 = new Image(this.getClass().getResourceAsStream(fmprod[n2]));
            Image newimg3 = new Image(this.getClass().getResourceAsStream(fmprod[n3]));
            product1img.setImage(newimg1);
            product2img.setImage(newimg2);
            product3img.setImage(newimg3);

        }
        else{
            Image newimg1 = new Image(this.getClass().getResourceAsStream(mprod[n1]));
            Image newimg2 = new Image(this.getClass().getResourceAsStream(mprod[n2]));
            Image newimg3 = new Image(this.getClass().getResourceAsStream(mprod[n3]));
            product1img.setImage(newimg1);
            product2img.setImage(newimg2);
            product3img.setImage(newimg3);

        }

    }
    public void goBack(ActionEvent actionEvent) throws IOException {

        FXMLLoader loader = new FXMLLoader(getClass().getResource("sample.fxml"));
        root = loader.load();
        stage = (Stage)((Node)actionEvent.getSource()).getScene().getWindow();
        scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }

    public void bgr1(MouseEvent event){
        product1img.setStyle("-fx-opacity: 1");
        product1img.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.8), 10, 0, 0, 0);");


    }
    public void bgr2(MouseEvent event){

        product2img.setStyle("-fx-opacity: 1");
        product2img.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.8), 10, 0, 0, 0);");


    }
    public void bgr3(MouseEvent event){

        product3img.setStyle("-fx-opacity: 1");
        product3img.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.8), 10, 0, 0, 0);");


    }
    public void hoverexit1(MouseEvent event){
        product1img.setStyle("-fx-opacity: 0.95");

    }
    public void hoverexit2(MouseEvent event){
        product2img.setStyle("-fx-opacity: 0.95");

    }
    public void hoverexit3(MouseEvent event){
        product3img.setStyle("-fx-opacity: 0.95");

    }


    public void OnClick(MouseEvent e) throws IOException, URISyntaxException {


        //        Hyperlink hp = new Hyperlink("https://www.amazon.in/s?k=Men+grooming+products&ref=nb_sb_noss_2");
//        hp.setGraphic(product1img);
//        hp.setGraphic(product2img);
        if(result.equals("Male")){
            Desktop.getDesktop().browse(new URL("https://www.amazon.in/mens-grooming-store/b?ie=UTF8&node=5122801031").toURI());
            Desktop.getDesktop().browse(new URL("https://www.amazon.in/s?k=Men+basketball+shoes&ref=nb_sb_noss_2").toURI());
        }
        else{
            Desktop.getDesktop().browse(new URL("https://www.amazon.in/s?k=Women+running+shoes&ref=nb_sb_noss").toURI());
            Desktop.getDesktop().browse(new URL("https://www.amazon.in/b?ie=UTF8&node=13165724031").toURI());

        }


    }


}
