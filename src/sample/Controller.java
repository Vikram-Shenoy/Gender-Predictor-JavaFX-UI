package sample;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.TextField;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;
import javafx.scene.image.ImageView;

import java.io.IOException;

public class Controller{

    @FXML
    TextField nameTextField;
    private Stage stage;
    private Scene scene;
    private Parent root;
    @FXML
    ImageView graph1,graph2;


    public void displayOp(ActionEvent actionEvent) throws IOException {
        String username = nameTextField.getText();
        FXMLLoader loader = new FXMLLoader(getClass().getResource("Scene2.fxml"));
        root = loader.load();
        Scene2 scene2Controller = loader.getController();

        scene2Controller.displayOutput(username);
        //root = FXMLLoader.load(getClass().getResource("Scene2.fxml"));
        stage = (Stage)((Node)actionEvent.getSource()).getScene().getWindow();
        scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }

    public void hoverenter1(MouseEvent event){
        graph1.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.8), 10, 0, 0, 0);");
    }
    public void hoverenter2(MouseEvent event){
        graph2.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.8), 10, 0, 0, 0);");
    }

    public void hoverexit1(MouseEvent event ) {
        graph1.setStyle("-fx-opacity:1");

    }
    public void hoverexit2(MouseEvent event ) {
        graph2.setStyle("-fx-opacity:1");

    }
    //    public Button yaho;
//    public void displayOtp(ActionEvent event) throws IOException {
//        String username = nameTextField.getText();
//
//        FXMLLoader loader = new FXMLLoader(getClass().getResource("Scene2.fxml"));
//        root = loader.load();
//        Scene2 scene2Controller = loader.getController();
//        scene2Controller.displayOutput(username);
//        //root = FXMLLoader.load(getClass().getResource("Scene2.fxml"));
//        stage = (Stage)((Node)event.getSource()).getScene().getWindow();
//        scene = new Scene(root);
//        stage.setScene(scene);
//        stage.show();
//
//    }
//    //    public void foto(MouseEvent event) throws  IOException {
////        System.out.println("link clicked");
////    }


}
